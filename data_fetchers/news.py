"""
News API fetcher for sentiment-driven alpha signals.
Uses NewsAPI.org free tier (100 requests/day).
"""
import logging
from typing import List, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
from config.settings import NEWS_API_KEY, NEWS_API_URL

logger = logging.getLogger(__name__)


class NewsApiFetcher:
    """Fetch news articles and headlines from NewsAPI.org."""

    BASE = NEWS_API_URL

    def _headers(self) -> dict:
        return {"X-Api-Key": NEWS_API_KEY}

    def get_everything(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 50,
    ) -> pd.DataFrame:
        """Search all articles matching a query."""
        if not NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set, skipping news fetch")
            return pd.DataFrame()

        if from_date is None:
            from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            resp = requests.get(
                f"{self.BASE}/everything",
                headers=self._headers(),
                params={
                    "q": query,
                    "from": from_date,
                    "to": to_date,
                    "language": language,
                    "sortBy": sort_by,
                    "pageSize": page_size,
                },
                timeout=15,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if not articles:
                return pd.DataFrame()

            rows = []
            for a in articles:
                rows.append({
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "content_snippet": (a.get("content", "") or "")[:500],
                })
            df = pd.DataFrame(rows)
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            return df
        except Exception as e:
            logger.error("NewsAPI fetch failed for '%s': %s", query, e)
            return pd.DataFrame()

    def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 20,
    ) -> pd.DataFrame:
        """Get top headlines for a category."""
        if not NEWS_API_KEY:
            return pd.DataFrame()
        try:
            resp = requests.get(
                f"{self.BASE}/top-headlines",
                headers=self._headers(),
                params={
                    "category": category,
                    "country": country,
                    "pageSize": page_size,
                },
                timeout=15,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            if not articles:
                return pd.DataFrame()

            rows = []
            for a in articles:
                rows.append({
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                })
            df = pd.DataFrame(rows)
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            return df
        except Exception as e:
            logger.error("NewsAPI headlines failed: %s", e)
            return pd.DataFrame()

    def get_news_for_assets(
        self, symbols: List[str], asset_type: str = "stock"
    ) -> pd.DataFrame:
        """Fetch news for a list of asset symbols."""
        all_news = []
        for sym in symbols:
            if asset_type == "crypto":
                query = f"{sym} cryptocurrency"
            elif asset_type == "prediction":
                query = sym  # use event title
            else:
                query = f"{sym} stock market"
            df = self.get_everything(query, page_size=10)
            if not df.empty:
                df["asset_symbol"] = sym
                df["asset_type"] = asset_type
                all_news.append(df)
        return pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
