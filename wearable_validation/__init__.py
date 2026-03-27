"""
wearable_validation
-------------------
Field HR validation protocol generator and analysis library.

Public API:
  generate_protocol()            — build a field test protocol
  compute_hrmax()                — Tanaka HRmax formula
  analyze_hr_validation()        — per-athlete agreement statistics
  analyze_group()                — multi-athlete group statistics
  analyze_by_intensity_bin()     — intensity-stratified accuracy
  check_hr_zone_coverage()       — HR zone coverage validation
  analyze_device_comparison()    — multi-device comparison ranked by MAPE
  generate_recommendation()      — use-case onboarding recommendation
  parse_combined_file()          — ingest single combined CSV/JSON
  parse_two_files()              — ingest two separate device files
  align_timeseries()             — timestamp-based linear interpolation
  format_report()                — plain-text per-athlete report
  format_group_report()          — plain-text group report
"""
from wearable_validation.models import (
    ProtocolParams,
    ProtocolStep,
    ProtocolStepBoundary,
    Protocol,
    TestRunMetadata,
    HRDataSeries,
    AnalysisReport,
    GroupAnalysisReport,
    IntensityBinResult,
    StepCoverageResult,
    CoverageReport,
    UseCaseRecommendation,
    OnboardingRecommendation,
    DeviceComparisonEntry,
    DeviceComparisonReport,
)
from wearable_validation.protocols import generate_protocol, compute_hrmax
from wearable_validation.io import parse_combined_file, parse_two_files, align_timeseries
from wearable_validation.analysis import (
    analyze_hr_validation,
    analyze_group,
    analyze_by_intensity_bin,
    check_hr_zone_coverage,
    analyze_device_comparison,
)
from wearable_validation.recommendation import generate_recommendation
from wearable_validation.report import format_report, format_group_report

__all__ = [
    # Models
    "ProtocolParams",
    "ProtocolStep",
    "ProtocolStepBoundary",
    "Protocol",
    "TestRunMetadata",
    "HRDataSeries",
    "AnalysisReport",
    "GroupAnalysisReport",
    "IntensityBinResult",
    "StepCoverageResult",
    "CoverageReport",
    "UseCaseRecommendation",
    "OnboardingRecommendation",
    "DeviceComparisonEntry",
    "DeviceComparisonReport",
    # Core functions
    "generate_protocol",
    "compute_hrmax",
    "parse_combined_file",
    "parse_two_files",
    "align_timeseries",
    "analyze_hr_validation",
    "analyze_group",
    "analyze_by_intensity_bin",
    "check_hr_zone_coverage",
    "analyze_device_comparison",
    "generate_recommendation",
    "format_report",
    "format_group_report",
]
