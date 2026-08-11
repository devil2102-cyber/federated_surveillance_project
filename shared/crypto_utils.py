import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_key_from_password(password: str, salt: bytes = b'federated_salt') -> bytes:
    """Generate a valid Fernet key from a string password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt byte data using symmetric Fernet encryption."""
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt byte data using symmetric Fernet encryption."""
    f = Fernet(key)
    return f.decrypt(encrypted_data)
