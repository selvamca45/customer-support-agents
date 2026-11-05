import importlib.metadata as md

for pkg in ["autogen-agentchat", "autogen-ext", "streamlit", "pandas",
            "python-dotenv", "openpyxl", "pydantic"]:
    try:
        print(f"OK       {pkg} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"MISSING  {pkg}")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
print("\nAutoGen imports work. Setup complete.")
