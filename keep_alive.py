"""Keep the Streamlit Community Cloud app awake.

A plain HTTP GET doesn't register as a viewer (Streamlit tracks websocket
connections), and a sleeping app requires clicking a wake-up button. This
script uses headless Chromium to do both.
"""

from playwright.sync_api import sync_playwright

URL = "https://fishvision-demo.streamlit.app/"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, timeout=90000, wait_until="domcontentloaded")

        # Sleeping apps show a "Yes, get this app back up!" button.
        for sel in (
            "button:has-text('back up')",
            "text=get this app back up",
            "text=wake",
        ):
            try:
                page.click(sel, timeout=5000)
                print(f"Clicked wake-up button via selector: {sel}")
                break
            except Exception:
                continue
        else:
            print("No wake-up button found — app is already awake")

        # Stay connected so the websocket session registers a real viewer.
        page.wait_for_timeout(60000)
        print("Page title:", page.title())
        browser.close()


if __name__ == "__main__":
    main()
