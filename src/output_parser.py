from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────
# Why Pydantic: enforces type safety, provides automatic validation,
# and gives us clean Python objects instead of raw dictionaries.

class KeyMetric(BaseModel):
    """A single financial metric extracted from the document."""
    name: str = Field(description="Name of the metric e.g. Revenue, Net Profit")
    value: str = Field(description="Value of the metric e.g. Rs 500 crore")
    period: str = Field(description="Time period e.g. FY2024, Q3 2025")
    trend: str = Field(description="Up, Down, or Stable compared to previous period")


class RiskFactor(BaseModel):
    """A risk factor identified in the document."""
    category: str = Field(description="Category e.g. Market Risk, Credit Risk, Regulatory")
    description: str = Field(description="Clear description of the risk")
    severity: str = Field(description="High, Medium, or Low")


class FinancialReport(BaseModel):
    """
    Structured financial analysis report.
    
    Why this structure: Financial analysts need consistent sections
    they can quickly scan. This mirrors actual analyst report formats
    used by investment banks and consulting firms.
    """
    document_title: str = Field(
        description="Title or name of the document being analysed"
    )
    executive_summary: str = Field(
        description="2-3 sentence summary of the document's key message"
    )
    key_metrics: List[KeyMetric] = Field(
        description="List of important financial metrics found in the document",
        default_factory=list
    )
    risk_factors: List[RiskFactor] = Field(
        description="List of risks identified in the document",
        default_factory=list
    )
    key_findings: List[str] = Field(
        description="List of 3-5 important findings from the document",
        default_factory=list
    )
    recommendations: List[str] = Field(
        description="List of 2-3 actionable recommendations based on findings",
        default_factory=list
    )
    data_gaps: List[str] = Field(
        description="Information that appears missing or unclear in the document",
        default_factory=list
    )
    confidence_score: float = Field(
        description="Confidence in analysis quality from 0.0 to 1.0 based on document clarity",
        ge=0.0,
        le=1.0
    )


# ── REPORT GENERATOR ─────────────────────────────────────────────────────────

def generate_structured_report(
    context: str,
    document_name: str
) -> FinancialReport:
    """
    Generate a structured financial analysis report from document context.
    
    WHY JsonOutputParser instead of StrOutputParser:
    - StrOutputParser returns raw text - unpredictable format
    - JsonOutputParser extracts JSON from LLM output and validates against schema
    - If LLM returns invalid JSON, parser raises clear error instead of silent failure
    
    WHY we inject format_instructions into the prompt:
    - The parser generates explicit instructions telling the LLM exactly
      what JSON structure to return
    - Without this, LLMs produce inconsistent output formats
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # JsonOutputParser generates format instructions automatically
    # from the Pydantic schema - no manual JSON template needed
    parser = JsonOutputParser(pydantic_object=FinancialReport)

    prompt = PromptTemplate(
        template="""You are a senior financial analyst.
Analyse the following document context and generate a structured report.
Document name: {document_name}

Context from document:
{context}

Generate a comprehensive financial analysis report.
If information for a field is not available in the context, use empty list or "Not available".
For confidence_score: use 0.9 if document is clear, 0.6 if partially clear, 0.3 if very limited info.

{format_instructions}""",
        input_variables=["context", "document_name"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "context": context,
            "document_name": document_name
        })

        # Result is a dict - convert to FinancialReport object
        if isinstance(result, dict):
            return FinancialReport(**result)
        return result

    except Exception as e:
        # Return minimal report on failure rather than crashing
        print(f"Parser error: {e}")
        return FinancialReport(
            document_title=document_name,
            executive_summary=f"Could not parse structured report: {str(e)}",
            key_metrics=[],
            risk_factors=[],
            key_findings=["Document parsing encountered an error"],
            recommendations=["Please try again with a clearer document"],
            data_gaps=["Full document analysis unavailable"],
            confidence_score=0.1
        )


def format_report_for_display(report: FinancialReport) -> str:
    """
    Format the structured report as readable text for terminal display.
    In the Streamlit app this will be replaced by formatted UI components.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"FINANCIAL ANALYSIS REPORT")
    lines.append(f"Document: {report.document_title}")
    lines.append("=" * 60)

    lines.append(f"\nEXECUTIVE SUMMARY")
    lines.append(report.executive_summary)

    if report.key_metrics:
        lines.append(f"\nKEY METRICS ({len(report.key_metrics)} found)")
        for m in report.key_metrics:
            lines.append(
                f"  • {m.name}: {m.value} ({m.period}) — Trend: {m.trend}"
            )

    if report.risk_factors:
        lines.append(f"\nRISK FACTORS ({len(report.risk_factors)} identified)")
        for r in report.risk_factors:
            lines.append(
                f"  [{r.severity}] {r.category}: {r.description}"
            )

    if report.key_findings:
        lines.append(f"\nKEY FINDINGS")
        for f in report.key_findings:
            lines.append(f"  • {f}")

    if report.recommendations:
        lines.append(f"\nRECOMMENDATIONS")
        for r in report.recommendations:
            lines.append(f"  → {r}")

    if report.data_gaps:
        lines.append(f"\nDATA GAPS")
        for g in report.data_gaps:
            lines.append(f"  ! {g}")

    lines.append(
        f"\nConfidence Score: {report.confidence_score:.0%}"
    )
    lines.append("=" * 60)

    return "\n".join(lines)