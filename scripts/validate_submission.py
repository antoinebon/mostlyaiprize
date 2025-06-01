#!/usr/bin/env python3
"""Validate competition submissions before upload."""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mostlyaiprize.evaluation.metrics import PrivacyUtilityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_submissions() -> List[Path]:
    """Find the latest submission files."""
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        return []
    
    # Find all submission files
    submission_files = list(submissions_dir.glob("*_submission_*.csv"))
    
    # Sort by modification time (newest first)
    submission_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return submission_files


def validate_file_format(submission_path: Path) -> Tuple[bool, List[str]]:
    """Validate submission file format."""
    issues = []
    
    try:
        # Check if file exists and is readable
        if not submission_path.exists():
            issues.append(f"File does not exist: {submission_path}")
            return False, issues
        
        # Check file size
        file_size_mb = submission_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # Arbitrary limit
            issues.append(f"File very large: {file_size_mb:.1f} MB")
        
        # Try to read CSV
        df = pd.read_csv(submission_path)
        
        # Basic format checks
        if len(df) == 0:
            issues.append("File is empty")
            return False, issues
        
        if len(df.columns) == 0:
            issues.append("No columns found")
            return False, issues
        
        # Check for completely null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            issues.append(f"Completely null columns: {null_columns}")
        
        # Check for suspicious patterns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_columns.append(col)
        
        if constant_columns:
            issues.append(f"Constant value columns: {constant_columns}")
        
        # Check data types
        if df.select_dtypes(include=['object']).empty and df.select_dtypes(include=['number']).empty:
            issues.append("No recognizable data types found")
        
        logger.info(f"✅ File format validation passed: {submission_path.name}")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Size: {file_size_mb:.1f} MB")
        logger.info(f"   Data types: {dict(df.dtypes.value_counts())}")
        
        return True, issues
        
    except Exception as e:
        issues.append(f"Failed to read file: {str(e)}")
        return False, issues


def validate_against_original(submission_path: Path, original_path: Path) -> Tuple[bool, List[str], dict]:
    """Validate submission against original dataset."""
    issues = []
    metrics = {}
    
    try:
        # Load both datasets
        submission_df = pd.read_csv(submission_path)
        original_df = pd.read_csv(original_path)
        
        # Shape validation
        if submission_df.shape[1] != original_df.shape[1]:
            issues.append(f"Column count mismatch: submission={submission_df.shape[1]}, original={original_df.shape[1]}")
        
        # Column name validation
        if not all(col in submission_df.columns for col in original_df.columns):
            missing_cols = set(original_df.columns) - set(submission_df.columns)
            issues.append(f"Missing columns: {missing_cols}")
        
        extra_cols = set(submission_df.columns) - set(original_df.columns)
        if extra_cols:
            issues.append(f"Extra columns: {extra_cols}")
        
        # Data type compatibility check
        for col in original_df.columns:
            if col in submission_df.columns:
                orig_dtype = original_df[col].dtype
                subm_dtype = submission_df[col].dtype
                
                # Check if numeric vs categorical mismatch
                if pd.api.types.is_numeric_dtype(orig_dtype) != pd.api.types.is_numeric_dtype(subm_dtype):
                    issues.append(f"Data type mismatch for column '{col}': original={orig_dtype}, submission={subm_dtype}")
        
        # Privacy check - no exact matches
        exact_matches = 0
        if len(issues) == 0:  # Only if format is correct
            for _, subm_row in submission_df.iterrows():
                if any((original_df == subm_row).all(axis=1)):
                    exact_matches += 1
                    if exact_matches >= 5:  # Stop after finding several
                        break
        
        if exact_matches > 0:
            issues.append(f"Privacy violation: Found {exact_matches} exact matches with original data")
        
        # Quality evaluation
        if len(issues) == 0:
            logger.info("🔍 Running quality evaluation...")
            evaluator = PrivacyUtilityEvaluator()
            
            # Ensure column order matches
            submission_df = submission_df[original_df.columns]
            
            metrics = evaluator.evaluate_privacy_utility(original_df, submission_df)
            
            # Quality thresholds
            utility_score = metrics.get('utility_score', 0)
            privacy_score = metrics.get('privacy_score', 0)
            competition_score = metrics.get('competition_score', 0)
            
            if utility_score < 0.5:
                issues.append(f"Low utility score: {utility_score:.3f} (< 0.5)")
            
            if privacy_score < 0.8:
                issues.append(f"Low privacy score: {privacy_score:.3f} (< 0.8)")
            
            if competition_score < 0.6:
                issues.append(f"Low competition score: {competition_score:.3f} (< 0.6)")
        
        return len(issues) == 0, issues, metrics
        
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        return False, issues, {}


def main():
    """Validate all submissions."""
    logger.info("🔍 Validating competition submissions...")
    
    # Find submission files
    submission_files = find_latest_submissions()
    
    if not submission_files:
        logger.error("❌ No submission files found in submissions/ directory")
        logger.info("💡 Run: hatch run dev:generate-submission")
        sys.exit(1)
    
    # Find original datasets
    data_dir = Path("data")
    original_files = list(data_dir.glob("*.csv"))
    
    if not original_files:
        logger.error("❌ No original datasets found in data/ directory")
        logger.info("💡 Run: hatch run dev:download-data")
        sys.exit(1)
    
    validation_results = []
    
    # Validate each submission
    for submission_path in submission_files[:2]:  # Validate latest 2 submissions
        logger.info(f"\n📋 Validating: {submission_path.name}")
        
        # Format validation
        format_ok, format_issues = validate_file_format(submission_path)
        
        if not format_ok:
            logger.error("❌ Format validation failed:")
            for issue in format_issues:
                logger.error(f"  • {issue}")
            validation_results.append((submission_path, False, format_issues, {}))
            continue
        
        # Find corresponding original dataset
        original_path = None
        if "challenge1" in submission_path.name:
            # Look for challenge1 dataset
            for pattern in ["challenge1", "dataset1"]:
                candidates = [f for f in original_files if pattern in f.name.lower()]
                if candidates:
                    original_path = candidates[0]
                    break
        elif "challenge2" in submission_path.name:
            # Look for challenge2 dataset
            for pattern in ["challenge2", "dataset2"]:
                candidates = [f for f in original_files if pattern in f.name.lower()]
                if candidates:
                    original_path = candidates[0]
                    break
        
        if original_path is None:
            # Use first/second file as fallback
            if "challenge1" in submission_path.name and len(original_files) >= 1:
                original_path = original_files[0]
            elif "challenge2" in submission_path.name and len(original_files) >= 2:
                original_path = original_files[1]
        
        if original_path is None:
            logger.warning(f"⚠️ Could not find original dataset for {submission_path.name}")
            validation_results.append((submission_path, False, ["Original dataset not found"], {}))
            continue
        
        logger.info(f"📊 Comparing against: {original_path.name}")
        
        # Content validation
        content_ok, content_issues, metrics = validate_against_original(submission_path, original_path)
        
        all_issues = format_issues + content_issues
        overall_ok = format_ok and content_ok
        
        if overall_ok:
            logger.info("✅ Validation passed!")
            if metrics:
                logger.info("📈 Quality metrics:")
                logger.info(f"  • Utility score: {metrics.get('utility_score', 0):.4f}")
                logger.info(f"  • Privacy score: {metrics.get('privacy_score', 0):.4f}")
                logger.info(f"  • Competition score: {metrics.get('competition_score', 0):.4f}")
        else:
            logger.error("❌ Validation failed:")
            for issue in all_issues:
                logger.error(f"  • {issue}")
        
        validation_results.append((submission_path, overall_ok, all_issues, metrics))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, ok, _, _ in validation_results if ok)
    total = len(validation_results)
    
    logger.info(f"Total submissions validated: {total}")
    logger.info(f"Passed validation: {passed}")
    logger.info(f"Failed validation: {total - passed}")
    
    if passed == total and total > 0:
        logger.info("\n🎉 All submissions are ready for upload!")
        logger.info("\n📤 Upload instructions:")
        logger.info("1. Go to https://www.mostlyaiprize.com/")
        logger.info("2. Navigate to submission portal")
        logger.info("3. Upload your CSV files")
        logger.info("4. Verify submission status")
    elif passed > 0:
        logger.info(f"\n✅ {passed} submission(s) ready for upload")
        logger.info(f"⚠️ {total - passed} submission(s) need fixes")
    else:
        logger.error("\n❌ No submissions passed validation!")
        logger.info("💡 Review the issues above and regenerate submissions")
        sys.exit(1)


if __name__ == "__main__":
    main()
