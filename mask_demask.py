from cryptography.fernet import Fernet

# Load the key
def load_key():
    return open('secret.key', 'rb').read()

# Mask the API key
def mask_api_key(api_key):
    key = load_key()
    fernet = Fernet(key)
    encrypted_key = fernet.encrypt(api_key.encode())
    return encrypted_key

# Demask the API key
def demask_api_key(encrypted_key):
    key = load_key()
    fernet = Fernet(key)
    decrypted_key = fernet.decrypt(encrypted_key).decode()
    return decrypted_key


