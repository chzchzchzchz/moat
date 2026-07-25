#!/usr/bin/env python3
"""
run_e2e_tests.py - Master CLI Test Runner for Project Antigravity E2E Test Suite
"""

import os
import sys
import time
import json
import argparse
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Master E2E Test Runner - Project Antigravity")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all", help="Test tier to execute (default: all)")
    parser.add_argument("--feature", choices=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "all"], default="all", help="Feature subset to filter (default: all)")
    parser.add_argument("--hardware", choices=["mock", "metal"], default="mock", help="Hardware execution mode (default: mock)")
    parser.add_argument("--json-report", type=str, default=None, help="Path to write JSON execution report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Determine test files based on tier
    test_dir = os.path.dirname(os.path.abspath(__file__))
    tier_file_map = {
        "1": os.path.join(test_dir, "test_tier1_features.py"),
        "2": os.path.join(test_dir, "test_tier2_boundaries.py"),
        "3": os.path.join(test_dir, "test_tier3_combinations.py"),
        "4": os.path.join(test_dir, "test_tier4_scenarios.py")
    }

    if args.tier == "all":
        target_files = list(tier_file_map.values())
    else:
        target_files = [tier_file_map[args.tier]]

    # Build pytest command-line arguments
    pytest_args = ["-q"]
    if args.verbose:
        pytest_args.append("-v")

    if args.feature != "all":
        pytest_args.extend(["-k", args.feature])

    pytest_args.extend(target_files)

    print("================================================================================")
    print("                PROJECT ANTIGRAVITY - E2E TEST SUITE RUNNER                     ")
    print("================================================================================")
    print(f"Target Tier     : {args.tier.upper()}")
    print(f"Target Feature  : {args.feature.upper()}")
    print(f"Hardware Mode   : {args.hardware.upper()}")
    print(f"Test Target(s)  : {', '.join([os.path.basename(f) for f in target_files])}")
    print("--------------------------------------------------------------------------------")

    start_time = time.time()
    
    # Plugin to record execution results for reporting
    class E2EReportPlugin:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.skipped = 0
            self.test_details = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                status = report.outcome.upper()
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
                elif report.skipped:
                    self.skipped += 1
                
                self.test_details.append({
                    "nodeid": report.nodeid,
                    "status": status,
                    "duration_sec": round(report.duration, 4)
                })

    report_plugin = E2EReportPlugin()
    exit_code = pytest.main(pytest_args, plugins=[report_plugin])
    elapsed_time = round(time.time() - start_time, 3)

    total_tests = report_plugin.passed + report_plugin.failed + report_plugin.skipped
    pass_rate = (report_plugin.passed / total_tests * 100.0) if total_tests > 0 else 0.0

    print("\n--------------------------------------------------------------------------------")
    print("                              E2E EXECUTION SUMMARY                              ")
    print("--------------------------------------------------------------------------------")
    print(f"Total Test Cases Executed : {total_tests}")
    print(f"Passed                    : {report_plugin.passed}")
    print(f"Failed                    : {report_plugin.failed}")
    print(f"Skipped                   : {report_plugin.skipped}")
    print(f"Pass Rate                 : {pass_rate:.1f}%")
    print(f"Total Duration            : {elapsed_time} s")
    print("================================================================================")

    # Generate JSON report if requested or default report
    report_data = {
        "title": "Project Antigravity E2E Test Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_mode": args.hardware,
        "tier_filter": args.tier,
        "feature_filter": args.feature,
        "total_executed": total_tests,
        "passed": report_plugin.passed,
        "failed": report_plugin.failed,
        "skipped": report_plugin.skipped,
        "pass_rate_pct": round(pass_rate, 2),
        "duration_seconds": elapsed_time,
        "tier_breakdown": {
            "tier1_feature_coverage": 35 if args.tier in ["1", "all"] else 0,
            "tier2_boundaries": 35 if args.tier in ["2", "all"] else 0,
            "tier3_combinations": 10 if args.tier in ["3", "all"] else 0,
            "tier4_scenarios": 7 if args.tier in ["4", "all"] else 0
        },
        "test_results": report_plugin.test_details
    }

    json_report_path = args.json_report or os.path.join(test_dir, "e2e_test_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Saved E2E JSON report to: {json_report_path}")

    # Return exit code 0 if all tests passed
    return 0 if exit_code == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
