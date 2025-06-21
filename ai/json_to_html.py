#!/usr/bin/env python3
"""
JSON to RTL HTML/Text Converter
Converts JSON files to nicely formatted RTL (Right-to-Left) HTML or text files.
Supports Arabic text and proper RTL formatting.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Union
import textwrap
from datetime import datetime

class JSONToRTLConverter:
    def __init__(self, json_file: str, output_format: str = 'html'):
        self.json_file = Path(json_file)
        self.output_format = output_format.lower()
        self.data = None
        
        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    def load_json(self) -> Dict[str, Any]:
        """Load and parse the JSON file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return self.data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise Exception(f"Error reading JSON file: {e}")
    
    def get_html_template(self) -> str:
        """Return the HTML template with RTL styling."""
        return '''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Document - {filename}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Noto Sans Arabic', 'Amiri', 'Times New Roman', serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
            color: #333;
            direction: rtl;
            text-align: right;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .title {{
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
            font-weight: bold;
        }}
        
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.1em;
            margin: 10px 0 0 0;
        }}
        
        .json-container {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            overflow-x: auto;
        }}
        
        .json-key {{
            color: #d73502;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
            font-size: 1.1em;
            border-right: 3px solid #3498db;
            padding-right: 10px;
            background: rgba(52, 152, 219, 0.1);
            padding: 8px 10px;
            border-radius: 3px;
        }}
        
        .json-value {{
            margin-right: 20px;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 3px;
            border: 1px solid #ecf0f1;
        }}
        
        .json-value.long-text {{
            line-height: 1.8;
            font-family: 'Noto Sans Arabic', serif;
            font-size: 1.05em;
            text-align: justify;
            padding: 15px;
            background: #fdfdfd;
        }}
        
        .json-value.short-text {{
            font-family: 'Courier New', monospace;
            background: #f1f2f6;
            color: #2f3542;
        }}
        
        .json-object {{
            border-right: 2px solid #95a5a6;
            margin-right: 15px;
            padding-right: 15px;
        }}
        
        .json-array {{
            border-right: 2px solid #e67e22;
            margin-right: 15px;
            padding-right: 15px;
        }}
        
        .array-index {{
            color: #e67e22;
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .level-0 {{ margin-right: 0; }}
        .level-1 {{ margin-right: 20px; }}
        .level-2 {{ margin-right: 40px; }}
        .level-3 {{ margin-right: 60px; }}
        .level-4 {{ margin-right: 80px; }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            font-size: 0.9em;
            color: #7f8c8d;
            text-align: center;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
        
        /* Better Arabic text rendering */
        .arabic-text {{
            font-family: 'Amiri', 'Noto Sans Arabic', serif;
            font-size: 1.1em;
            line-height: 1.8;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            body {{ padding: 10px; }}
            .container {{ padding: 15px; }}
            .title {{ font-size: 2em; }}
            .level-1, .level-2, .level-3, .level-4 {{ margin-right: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">مستند JSON</h1>
            <p class="subtitle">الملف: {filename} | تاريخ التحويل: {date}</p>
        </div>
        
        <div class="json-container">
            {content}
        </div>
        
        <div class="footer">
            تم إنشاؤه بواسطة محول JSON إلى HTML | Generated by JSON to HTML Converter
        </div>
    </div>
</body>
</html>'''
    
    def detect_arabic_text(self, text: str) -> bool:
        """Detect if text contains Arabic characters."""
        if not isinstance(text, str):
            return False
        
        arabic_chars = any('\u0600' <= char <= '\u06FF' for char in text)
        return arabic_chars
    
    def format_json_to_html(self, data: Any, level: int = 0) -> str:
        """Convert JSON data to formatted HTML."""
        html_parts = []
        level_class = f"level-{min(level, 4)}"
        
        if isinstance(data, dict):
            html_parts.append(f'<div class="json-object {level_class}">')
            for key, value in data.items():
                # Format key
                key_html = f'<div class="json-key">{self.escape_html(str(key))}</div>'
                html_parts.append(key_html)
                
                # Format value
                value_html = self.format_json_to_html(value, level + 1)
                html_parts.append(f'<div class="json-value">{value_html}</div>')
            
            html_parts.append('</div>')
            
        elif isinstance(data, list):
            html_parts.append(f'<div class="json-array {level_class}">')
            for i, item in enumerate(data):
                # Array index
                index_html = f'<div class="array-index">[{i}]</div>'
                html_parts.append(index_html)
                
                # Array item
                item_html = self.format_json_to_html(item, level + 1)
                html_parts.append(f'<div class="json-value">{item_html}</div>')
            
            html_parts.append('</div>')
            
        else:
            # Handle primitive values
            text = str(data)
            escaped_text = self.escape_html(text)
            
            if len(text) > 100:
                # Long text
                css_class = "long-text"
                if self.detect_arabic_text(text):
                    css_class += " arabic-text"
                return f'<div class="{css_class}">{escaped_text}</div>'
            else:
                # Short text
                return f'<span class="short-text">{escaped_text}</span>'
        
        return ''.join(html_parts)
    
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def generate_html(self, output_file: str = None):
        """Generate RTL HTML from JSON data."""
        if not self.data:
            self.load_json()
        
        if not output_file:
            output_file = self.json_file.with_suffix('.html')
        
        # Format JSON content
        content = self.format_json_to_html(self.data)
        
        # Get template and fill it
        template = self.get_html_template()
        html_content = template.format(
            filename=self.json_file.name,
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            content=content
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML file generated: {output_file}")
        print(f"Open with: xdg-open {output_file}")
    
    def generate_text(self, output_file: str = None):
        """Generate RTL text file from JSON data."""
        if not self.data:
            self.load_json()
        
        if not output_file:
            output_file = self.json_file.with_suffix('.txt')
        
        # Format the JSON data
        formatted_text = self._format_json_for_text(self.data)
        
        # Add header
        header = f"مستند JSON: {self.json_file.name}\n"
        header += f"JSON Document: {self.json_file.name}\n"
        header += "=" * 60 + "\n"
        header += f"تاريخ التحويل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Conversion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Write to file with UTF-8 encoding for proper Arabic support
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(formatted_text)
        
        print(f"Text file generated: {output_file}")
    
    def _format_json_for_text(self, data: Any, level: int = 0) -> str:
        """Format JSON data as readable text."""
        indent = "  " * level
        result = ""
        
        if isinstance(data, dict):
            for key, value in data.items():
                result += f"{indent}• {key}:\n"
                if isinstance(value, (dict, list)):
                    result += self._format_json_for_text(value, level + 1)
                else:
                    # Add proper text wrapping for long content
                    text_value = str(value)
                    if len(text_value) > 80:
                        wrapped = textwrap.fill(text_value, width=80, 
                                              initial_indent=indent + "  ",
                                              subsequent_indent=indent + "  ")
                        result += f"{wrapped}\n"
                    else:
                        result += f"{indent}  {text_value}\n"
                result += "\n"
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                result += f"{indent}[{i}]:\n"
                result += self._format_json_for_text(item, level + 1)
                result += "\n"
                
        else:
            result += f"{indent}{data}\n"
        
        return result
    
    def convert(self, output_file: str = None):
        """Convert JSON to the specified format."""
        if self.output_format == 'html':
            self.generate_html(output_file)
        elif self.output_format == 'text' or self.output_format == 'txt':
            self.generate_text(output_file)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON files to RTL-formatted HTML or text files'
    )
    parser.add_argument(
        'input_file',
        help='Input JSON file path'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['html', 'text', 'txt'],
        default='html',
        help='Output format (default: html)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        converter = JSONToRTLConverter(args.input_file, args.format)
        converter.convert(args.output)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()