# app/reporting.py
"""
Robust Report Generation Module for FinanceIQ

Features:
- JSON, Excel, and CSV report generation
- Input validation
- Safe Excel sheet naming
- Optional report cleanup
- Logging support
- Configurable row limits
- Graceful dependency handling
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXCEL_ROWS = 100_000
DEFAULT_RETENTION_DAYS = 30


class ReportGenerationError(Exception):
    """Raised when report generation fails."""


class ReportGenerator:
    """
    Generates formal financial reports.

    Supports:
    - JSON reports
    - Excel reports
    - CSV exports
    - Financial summaries
    - Anomaly reports
    - Trend reports
    - Evaluation reports
    """

    REQUIRED_FINANCIAL_COLUMNS = {"amount"}

    def __init__(
        self,
        output_dir: str = "reports",
        max_excel_rows: int = DEFAULT_MAX_EXCEL_ROWS,
    ):
        self.output_dir = Path(output_dir)
        self.max_excel_rows = max_excel_rows

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        allow_empty: bool = False,
    ) -> None:

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if not allow_empty and df.empty:
            raise ValueError("DataFrame is empty")

        required_columns = required_columns or []

        missing = [c for c in required_columns if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================
    # EXCEL HELPERS
    # =========================================================

    @staticmethod
    def _safe_sheet_name(
        name: str,
        used_names: Optional[set] = None,
    ) -> str:
        """
        Create Excel-safe sheet names.
        """

        used_names = used_names or set()

        # remove invalid chars
        name = re.sub(r'[\[\]\:\*\?\/\\]', '_', str(name))

        # trim length
        name = name[:31]

        if not name:
            name = "Sheet"

        original = name
        counter = 1

        while name in used_names:
            suffix = f"_{counter}"

            max_len = 31 - len(suffix)

            name = original[:max_len] + suffix

            counter += 1

        used_names.add(name)

        return name

    def _get_excel_engine(self) -> str:
        """
        Return available Excel engine.
        """

        try:
            import openpyxl  # noqa
            return "openpyxl"
        except ImportError:
            try:
                import xlsxwriter  # noqa
                return "xlsxwriter"
            except ImportError:
                raise ReportGenerationError(
                    "No Excel writer available. Install openpyxl or xlsxwriter."
                )

    # =========================================================
    # FILE HELPERS
    # =========================================================

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _safe_user(user: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user))

    def _limit_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > self.max_excel_rows:
            logger.warning(
                "DataFrame exceeds Excel row limit (%s). Truncating.",
                self.max_excel_rows
            )
            return df.head(self.max_excel_rows)
        return df

    # =========================================================
    # EXPORT HELPERS
    # =========================================================

    def _write_json(self, data: Dict, path: Path) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.exception("Failed to write JSON report")
            raise ReportGenerationError(str(e))

    def _write_csv(self, df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            logger.exception("Failed to write CSV report")
            raise ReportGenerationError(str(e))

    # =========================================================
    # FINANCIAL SUMMARY
    # =========================================================

    def generate_financial_summary(
        self,
        df: pd.DataFrame,
        user: str,
        analysis: Optional[Dict] = None,
    ) -> Tuple[str, str]:

        self._validate_dataframe(
            df,
            required_columns=["amount"],
        )

        from app.financial_ai import full_analysis
        from app.health import (
            calculate_health_score,
            health_insights,
        )
        from app.predictor import predict_next_spending

        user = self._safe_user(user)
        timestamp = self._timestamp()

        logger.info("Generating financial summary for %s", user)

        # reuse existing analysis if provided
        if analysis is None:
            analysis = full_analysis(df)

        health_score = calculate_health_score(df)
        insights = health_insights(df)
        next_spend = predict_next_spending(df)

        date_range = {
            "start": None,
            "end": None,
        }

        if "date" in df.columns:
            try:
                dates = pd.to_datetime(df["date"], errors="coerce")
                date_range = {
                    "start": dates.min().isoformat() if not dates.isna().all() else None,
                    "end": dates.max().isoformat() if not dates.isna().all() else None,
                }
            except Exception:
                logger.warning("Failed to parse date range")

        simulation = analysis.get("optimization", {}).get("simulation", {})

        difference = simulation.get("difference", 0)

        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user": user,
                "framework_version": "2.0.0",
                "total_records": len(df),
                "date_range": date_range,
            },
            "financial_health": {
                "score": health_score,
                "insights": insights,
            },
            "metrics": analysis.get("metrics", {}),
            "forecast": {
                "next_expected_spend": next_spend,
                "trend": analysis.get("forecast", {}).get("trend"),
                "forecast_values": analysis.get("forecast", {}).get(
                    "forecast",
                    []
                )[:5],
            },
            "optimization": {
                "difference": difference,
                "projected_savings": (
                    abs(difference)
                    if difference < 0
                    else 0
                ),
                "recommendations": analysis.get(
                    "optimization",
                    {}
                ).get("recommendations", []),
            },
            "risks": analysis.get("risks", []),
            "insights": analysis.get("insights", []),
        }

        json_path = self.output_dir / f"summary_{user}_{timestamp}.json"
        excel_path = self.output_dir / f"summary_{user}_{timestamp}.xlsx"

        self._write_json(report, json_path)

        # Excel generation
        engine = self._get_excel_engine()

        try:
            used_names = set()

            with pd.ExcelWriter(excel_path, engine=engine) as writer:

                # Transactions
                tx_sheet = self._safe_sheet_name(
                    "Transactions",
                    used_names,
                )

                self._limit_rows(df).to_excel(
                    writer,
                    sheet_name=tx_sheet,
                    index=False,
                )

                # Summary
                summary_rows = [
                    {
                        "Metric": "Health Score",
                        "Value": health_score,
                    },
                    {
                        "Metric": "Next Spend",
                        "Value": next_spend,
                    },
                    {
                        "Metric": "Projected Savings",
                        "Value": report["optimization"]["projected_savings"],
                    },
                ]

                pd.DataFrame(summary_rows).to_excel(
                    writer,
                    sheet_name=self._safe_sheet_name(
                        "Summary",
                        used_names,
                    ),
                    index=False,
                )

                # Category breakdown
                category_breakdown = (
                    analysis.get("metrics", {})
                    .get("category_breakdown", {})
                )

                if category_breakdown:
                    pd.DataFrame(
                        list(category_breakdown.items()),
                        columns=["Category", "Amount"],
                    ).to_excel(
                        writer,
                        sheet_name=self._safe_sheet_name(
                            "Categories",
                            used_names,
                        ),
                        index=False,
                    )

                # Risks
                risks = analysis.get("risks", [])

                if risks:
                    pd.DataFrame(
                        {"Risk": risks}
                    ).to_excel(
                        writer,
                        sheet_name=self._safe_sheet_name(
                            "Risks",
                            used_names,
                        ),
                        index=False,
                    )

                # Insights
                ai_insights = analysis.get("insights", [])

                if ai_insights:
                    pd.DataFrame(
                        {"Insight": ai_insights}
                    ).to_excel(
                        writer,
                        sheet_name=self._safe_sheet_name(
                            "Insights",
                            used_names,
                        ),
                        index=False,
                    )

        except Exception as e:
            logger.exception("Excel report generation failed")
            raise ReportGenerationError(str(e))

        logger.info("Financial summary generated successfully")

        return str(json_path), str(excel_path)

    # =========================================================
    # ANOMALY REPORT
    # =========================================================

    def generate_anomaly_report(
        self,
        anomalies_df: pd.DataFrame,
    ) -> Optional[str]:

        self._validate_dataframe(
            anomalies_df,
            allow_empty=True,
        )

        if anomalies_df.empty:
            logger.warning("No anomalies found")
            return None

        timestamp = self._timestamp()

        report_path = self.output_dir / f"anomalies_{timestamp}.xlsx"

        engine = self._get_excel_engine()

        try:
            with pd.ExcelWriter(report_path, engine=engine) as writer:

                self._limit_rows(anomalies_df).to_excel(
                    writer,
                    sheet_name="All_Anomalies",
                    index=False,
                )

                summary = {
                    "Total Anomalies": len(anomalies_df),
                }

                if "amount" in anomalies_df.columns:
                    summary["Average Amount"] = (
                        anomalies_df["amount"].mean()
                    )

                pd.DataFrame(
                    list(summary.items()),
                    columns=["Metric", "Value"],
                ).to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False,
                )

        except Exception as e:
            logger.exception("Failed to generate anomaly report")
            raise ReportGenerationError(str(e))

        return str(report_path)

    # =========================================================
    # TREND REPORT
    # =========================================================

    def generate_trend_report(
        self,
        trends: Dict,
    ) -> Optional[str]:

        if not trends:
            raise ValueError("Trend data is empty")

        timestamp = self._timestamp()

        report_path = self.output_dir / f"trends_{timestamp}.xlsx"

        engine = self._get_excel_engine()

        try:
            used_names = set()

            with pd.ExcelWriter(report_path, engine=engine) as writer:

                for key, value in trends.items():

                    sheet_name = self._safe_sheet_name(
                        str(key),
                        used_names,
                    )

                    if isinstance(value, pd.DataFrame):

                        self._limit_rows(value).to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=False,
                        )

                    elif isinstance(value, pd.Series):

                        value.to_frame(name=key).to_excel(
                            writer,
                            sheet_name=sheet_name,
                        )

                    elif isinstance(value, dict):

                        pd.DataFrame(
                            list(value.items()),
                            columns=["Key", "Value"],
                        ).to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=False,
                        )

        except Exception as e:
            logger.exception("Trend report generation failed")
            raise ReportGenerationError(str(e))

        return str(report_path)

    # =========================================================
    # EVALUATION REPORT
    # =========================================================

    def generate_evaluation_report(
        self,
        evaluation_results: Dict,
    ) -> Tuple[str, str]:

        if not isinstance(evaluation_results, dict):
            raise ValueError("evaluation_results must be a dictionary")

        timestamp = self._timestamp()

        json_path = self.output_dir / f"evaluation_{timestamp}.json"
        excel_path = self.output_dir / f"evaluation_{timestamp}.xlsx"

        self._write_json(evaluation_results, json_path)

        tests = evaluation_results.get("tests", [])

        engine = self._get_excel_engine()

        try:
            used_names = set()

            with pd.ExcelWriter(excel_path, engine=engine) as writer:

                if not tests:

                    pd.DataFrame(
                        [{"Message": "No evaluation tests found"}]
                    ).to_excel(
                        writer,
                        sheet_name="Summary",
                        index=False,
                    )

                for test in tests:

                    test_name = test.get(
                        "test_name",
                        "Unnamed_Test",
                    )

                    sheet_name = self._safe_sheet_name(
                        test_name,
                        used_names,
                    )

                    metrics = test.get("metrics", {})

                    rows = [
                        {
                            "Metric": k,
                            "Value": v,
                        }
                        for k, v in metrics.items()
                    ]

                    pd.DataFrame(rows).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                    )

        except Exception as e:
            logger.exception("Evaluation report generation failed")
            raise ReportGenerationError(str(e))

        return str(json_path), str(excel_path)

    # =========================================================
    # MONTHLY REPORT
    # =========================================================

    def generate_monthly_analysis_report(
        self,
        user_name: str,
        year: int,
        month: int,
        daily_items_df: pd.DataFrame,
    ) -> str:

        self._validate_dataframe(
            daily_items_df,
            required_columns=[
                "type",
                "amount",
                "category",
                "description",
            ],
        )

        user_name = self._safe_user(user_name)

        expenses = daily_items_df[
            daily_items_df["type"] == "expense"
        ]

        sales = daily_items_df[
            daily_items_df["type"] == "sale"
        ]

        total_expenses = float(expenses["amount"].sum())
        total_sales = float(sales["amount"].sum())

        report = {
            "user": user_name,
            "year": year,
            "month": month,
            "total_expenses": total_expenses,
            "total_sales": total_sales,
            "net": total_sales - total_expenses,
            "top_expense_categories": (
                expenses.groupby("category")["amount"]
                .sum()
                .to_dict()
            ),
            "top_selling_items": (
                sales.groupby("description")["amount"]
                .sum()
                .nlargest(5)
                .to_dict()
            ),
        }

        timestamp = self._timestamp()

        json_path = (
            self.output_dir /
            f"monthly_{user_name}_{year}_{month}_{timestamp}.json"
        )

        self._write_json(report, json_path)

        return str(json_path)

    # =========================================================
    # REPORT LISTING
    # =========================================================

    def list_reports(self) -> pd.DataFrame:

        reports = []

        if not self.output_dir.exists():
            return pd.DataFrame()

        for file in self.output_dir.iterdir():

            if not file.is_file():
                continue

            reports.append({
                "filename": file.name,
                "type": file.suffix.replace(".", "").upper(),
                "size_kb": round(file.stat().st_size / 1024, 2),
                "modified": datetime.fromtimestamp(
                    file.stat().st_mtime
                ).isoformat(),
            })

        return pd.DataFrame(reports)

    # =========================================================
    # CLEANUP
    # =========================================================

    def cleanup_old_reports(
        self,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> int:
        """
        Delete reports older than retention_days.
        """

        deleted = 0

        cutoff = datetime.utcnow() - timedelta(
            days=retention_days
        )

        for file in self.output_dir.iterdir():

            if not file.is_file():
                continue

            modified = datetime.utcfromtimestamp(
                file.stat().st_mtime
            )

            if modified < cutoff:
                try:
                    file.unlink()
                    deleted += 1
                except Exception:
                    logger.exception(
                        "Failed to delete old report: %s",
                        file,
                    )

        logger.info("Deleted %s old reports", deleted)

        return deleted