# For verification / judging, use OpenAI
import getpass
from openai import OpenAI

# Initalise variables
apiKey = None
client = None

# If api key is to be used
if(input("Use api key? Y/N \n") == "Y"):
    # Use openAI key to access datasets
    apiKey = getpass.getpass(prompt="Enter openAI api key:")
    # OpenAI client using the key
    client = OpenAI(api_key=apiKey)