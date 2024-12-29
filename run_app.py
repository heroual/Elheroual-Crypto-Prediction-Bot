import streamlit.web.cli as stcli
import os
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "src/app.py", "--server.port=8502", "--server.address=localhost", "--browser.serverAddress=localhost"]
    sys.exit(stcli.main())
