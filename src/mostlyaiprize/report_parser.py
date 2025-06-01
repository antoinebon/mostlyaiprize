"""Parser for MOSTLY AI HTML quality reports."""

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


class ReportParser:
    """Parser for extracting metrics from MOSTLY AI HTML quality reports."""
    
    def __init__(self, html_content: str) -> None:
        """Initialize parser with HTML content.
        
        Args:
            html_content: Raw HTML content from MOSTLY AI report
        """
        self._soup: BeautifulSoup = BeautifulSoup(html_content, 'html.parser')
    
    @classmethod
    def from_file(cls, file_path: str | Path) -> 'ReportParser':
        """Create parser from HTML file.
        
        Args:
            file_path: Path to HTML report file
            
        Returns:
            Configured ReportParser instance
        """
        content: str = Path(file_path).read_text(encoding='utf-8')
        return cls(content)
    
    def extract_metrics(self) -> dict[str, float]:
        """Extract key metrics from the report.
        
        Returns:
            Dictionary containing accuracy, DCR share, and NNDR ratio
            
        Raises:
            ValueError: If required metrics cannot be extracted
        """
        metrics: dict[str, float] = {}
        
        # Extract overall accuracy (main percentage in first result box)
        accuracy_element = self._soup.find('div', class_='result-box-large-title')
        if accuracy_element:
            accuracy_text: str = accuracy_element.get_text(strip=True)
            accuracy_match = re.search(r'(\d+\.?\d*)%', accuracy_text)
            if accuracy_match:
                metrics['accuracy'] = float(accuracy_match.group(1))
        
        # Extract DCR Share and NNDR Ratio from table rows
        table_rows = self._soup.find_all('tr')
        
        for row in table_rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                metric_name: str = cells[0].get_text(strip=True)
                metric_value_cell = cells[-1]  # Last cell contains the value
                
                if metric_name == 'DCR Share':
                    dcr_text: str = metric_value_cell.get_text(strip=True)
                    dcr_match = re.search(r'(\d+\.?\d*)%', dcr_text)
                    if dcr_match:
                        metrics['dcr_share'] = float(dcr_match.group(1))
                
                elif metric_name == 'NNDR Ratio':
                    nndr_text: str = metric_value_cell.get_text(strip=True)
                    nndr_match = re.search(r'(\d+\.?\d*)', nndr_text)
                    if nndr_match:
                        metrics['nndr_ratio'] = float(nndr_match.group(1))
        
        # Validate that we got all required metrics
        required_metrics = ['accuracy', 'dcr_share', 'nndr_ratio']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if missing_metrics:
            raise ValueError(f"Failed to extract required metrics: {missing_metrics}")
        
        return metrics


def parse_report_metrics(html_content: str) -> dict[str, float]:
    """Parse key metrics from MOSTLY AI HTML report.
    
    Args:
        html_content: Raw HTML content from MOSTLY AI report
        
    Returns:
        Dictionary with accuracy, DCR share, and NNDR ratio
    """
    parser = ReportParser(html_content)
    return parser.extract_metrics()


__all__ = ["ReportParser", "parse_report_metrics"]
