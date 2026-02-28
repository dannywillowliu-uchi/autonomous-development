"""External intelligence subsystem for monitoring AI/agent ecosystem developments."""

from mission_control.intelligence.evaluator import evaluate_findings, generate_proposals
from mission_control.intelligence.models import AdaptationProposal, Finding, IntelSource
from mission_control.intelligence.scanner import IntelReport, IntelScanner, run_scan
from mission_control.intelligence.sources import scan_arxiv, scan_github, scan_hackernews

__all__ = [
	"AdaptationProposal",
	"Finding",
	"IntelReport",
	"IntelScanner",
	"IntelSource",
	"evaluate_findings",
	"generate_proposals",
	"run_scan",
	"scan_arxiv",
	"scan_github",
	"scan_hackernews",
]
