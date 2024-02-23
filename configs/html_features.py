HTML_WRAPPER = """
<div style="overflow-x: auto; border: 1px solid #e6e9ef;">{}</div>
"""

# def set_image(url):
#     HTML_BACKGROUND_IMG = f"<style> body {{ background-image: url('{url}'); background-size: 100%; background-repeat: no-repeat; }} </style>"
#     return HTML_BACKGROUND_IMG


def set_image(url):
    HTML_BACKGROUND_IMG = f"""
    <style>
        body {{
            background-image: url('{url}');
            background-size: 100%;
            background-repeat: no-repeat;
        }}
    </style>
    """
    return HTML_BACKGROUND_IMG

