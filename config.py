from cryptography.fernet import Fernet

# Generate a key for encryption and decryption
# You must keep this key safe and not expose it in your codebase
key = Fernet.generate_key()

# Store the key in a safe place, or environment variable
with open('secret.key', 'wb') as key_file:
    key_file.write(key)
