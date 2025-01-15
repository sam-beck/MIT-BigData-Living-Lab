# For verification / judging, use OpenAI
import getpass
from openai import OpenAI

# Use openAI key to access datasets
apiKey = getpass.getpass(prompt="Enter openAI api key:")
# OpenAI client using the key
client = OpenAI(api_key=apiKey)