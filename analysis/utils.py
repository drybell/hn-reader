import re
import html

class Cleaners:

    @staticmethod
    def strip(text : str):
        """
        Strip HTML tags from text while preserving content.

        Converts HTML entities to their corresponding characters
        and removes all HTML tags, keeping the text content.

        Args:
            text: The HTML string to strip, or None

        Returns:
            Plain text with HTML removed and entities decoded
        """
        # First decode HTML entities (e.g., &quot; -> ", &#x27; -> ')
        decoded = html.unescape(text)

        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', decoded)

        # Strip Markdown formatting
        # Links: [text](url) -> text
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
        # Images: ![alt](url) -> alt
        clean = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', clean)
        # Bold/italic: **text** or __text__ or *text* or _text_ -> text
        clean = re.sub(r'\*\*([^\*]+)\*\*', r'\1', clean)
        clean = re.sub(r'__([^_]+)__', r'\1', clean)
        clean = re.sub(r'\*([^\*]+)\*', r'\1', clean)
        clean = re.sub(r'_([^_]+)_', r'\1', clean)
        # Inline code: `code` -> code
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        # Headers: # Header -> Header
        clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)
        # Strikethrough: ~~text~~ -> text
        clean = re.sub(r'~~([^~]+)~~', r'\1', clean)
        # Code blocks: ```code``` or ~~~code~~~ -> code
        clean = re.sub(r'```[^\n]*\n(.*?)```', r'\1', clean, flags=re.DOTALL)
        clean = re.sub(r'~~~[^\n]*\n(.*?)~~~', r'\1', clean, flags=re.DOTALL)
        # Blockquotes: > text -> text
        clean = re.sub(r'^>\s+', '', clean, flags=re.MULTILINE)

        # Clean up any extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean
