"""
News API fetcher with resilience, deduplication, and source credibility.
"""
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set
import pandas as pd
from config.settings import NEWS_API_CONFIG
from config.resilience import ResilientClient

logger = logging.getLogger(__name__)

# Source credibility tiers (institutional news sources weighted higher)
SOURCE_CREDIBILITY: dict = {
    "Reuters": 1.0, "Bloomberg": 1.0, "Financial Times": 0.95,
    "The Wall Street Journal": 0.95, "CNBC": 0.85, "BBC News": 0.85,
    "The New York Times": 0.85, "Associated Press": 0.9,
    "MarketWatch": 0.8, "Barron's": 0.85, "The Economist": 0.9,
    "Yahoo Finance": 0.7, "Seeking Alpha": 0.6, "Benzinga": 0.65,
    "Investopedia": 0.6, "CoinDesk": 0.7, "The Block": 0.7,
    "Decrypt": 0.6, "CoinTelegraph": 0.55,
}
DEFAULT_CREDIBILITY = 0.5


class NewsApiFetcher:
    """Fetch news articles from NewsAPI.org with dedup and credibility."""

    def __init__(self):
        self.available = bool(NEWS_API_CONFIG.api_key)
        if self.available:
            self.client = ResilientClient(
                name="newsapi", base_url=NEWS_API_CONFIG.base_url,
                timeout=NEWS_API_CONFIG.timeout_s,
                retry_policy=NEWS_API_CONFIG.retry_policy,
                rate_limit_per_min=NEWS_API_CONFIG.rate_limit_per_min,
                default_headers={"X-Api-Key": NEWS_API_CONFIG.api_key},
            )
        self._seen_hashes: Set[str] = set()

    def _article_hash(self, title: str, source: str) -> str:
        return hashlib.md5(f"{title}|{source}".encode()).hexdigest()

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["_hash"] = df.apply(
            lambda r: self._article_hash(str(r.get("title", "")),
                                          str(r.get("source", ""))), axis=1
        )
        new_mask = ~df["_hash"].isin(self._seen_hashes)
        self._seen_hashes |= set(df["_hash"])
        return df[new_mask].drop(columns=["_hash"])

    def get_everything(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 50,
    ) -> pd.DataFrame:
        if not self.available:
            logger.debug("NEWS_API_KEY not set, skipping news fetch")
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        if from_date is None:
            from_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = now.strftime("%Y-%m-%d")

        resp = self.client.get("everything", params={
            "q": query, "from": from_date, "to": to_date,
            "language": language, "sortBy": sort_by, "pageSize": page_size,
        })
        if resp is None:
            return pd.DataFrame()

        try:
            body = resp.json()
        except ValueError:
            return pd.DataFrame()

        if body.get("status") != "ok":
            logger.error("NewsAPI error: %s", body.get("message", "unknown"))
            return pd.DataFrame()

        articles = body.get("articles", [])
        if not articles:
            return pd.DataFrame()

        rows = []
        for a in articles:
            source_name = a.get("source", {}).get("name", "")
            rows.append({
                "title": str(a.get("title", "") or ""),
                "description": str(a.get("description", "") or ""),
                "source": source_name,
                "url": str(a.get("url", "") or ""),
                "published_at": a.get("publishedAt", ""),
                "content_snippet": (str(a.get("content", "") or ""))[:500],
                "source_credibility": SOURCE_CREDIBILITY.get(
                    source_name, DEFAULT_CREDIBILITY
                ),
            })
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = self._deduplicate(df)
        return df

    def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 20,
    ) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        resp = self.client.get("top-headlines", params={
            "category": category, "country": country, "pageSize": page_size,
        })
        if resp is None:
            return pd.DataFrame()

        try:
            body = resp.json()
        except ValueError:
            return pd.DataFrame()

        articles = body.get("articles", [])
        if not articles:
            return pd.DataFrame()

        rows = []
        for a in articles:
            source_name = a.get("source", {}).get("name", "")
            rows.append({
                "title": str(a.get("title", "") or ""),
                "description": str(a.get("description", "") or ""),
                "source": source_name,
                "url": str(a.get("url", "") or ""),
                "published_at": a.get("publishedAt", ""),
                "source_credibility": SOURCE_CREDIBILITY.get(
                    source_name, DEFAULT_CREDIBILITY
                ),
            })
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        return df

    def get_news_for_assets(
        self, symbols: List[str], asset_type: str = "stock"
    ) -> pd.DataFrame:
        all_news = []
        for sym in symbols:
            if asset_type == "crypto":
                query = f"{sym} cryptocurrency"
            elif asset_type == "prediction":
                query = sym
            else:
                query = f"{sym} stock market"
            df = self.get_everything(query, page_size=10)
            if not df.empty:
                df["asset_symbol"] = sym
                df["asset_type"] = asset_type
                all_news.append(df)
        return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
