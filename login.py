from huggingface_hub import login
from hf_token import get_token # This module is in .gitignore to not publish the token.

login(get_token(), add_to_git_credential=True) # Log in to HuggingFace Hub using a private token