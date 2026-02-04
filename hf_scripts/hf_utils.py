from huggingface_hub import login, whoami
from huggingface_hub.utils import LocalTokenNotFoundError


def authenticate_hf(token=None):
    '''Authenticate with Hugging Face. Args: token (str|None). Returns: None.'''
    try:
        print("Huggingface authentication. Who am I?:", whoami())
    except LocalTokenNotFoundError:
        login(token=token)
        print("Authentication complete.")
        print("Huggingface authentication. Who am I?:", whoami())
