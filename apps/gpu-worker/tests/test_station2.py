"""Trạm 2 (Station 2) — Kiểm thử xác thực JWT BẤT ĐỐI XỨNG (ES256).

Chứng minh: Worker chỉ giữ PUBLIC key, xác thực token do Gateway ký bằng PRIVATE key,
fail-closed khi thiếu khóa, và miễn nhiễm với tấn công alg-confusion (HS256).
Không dùng secret đối xứng hardcoded ở bất cứ đâu.
"""
import base64
import hashlib
import hmac
import json
import pytest
from datetime import datetime, timezone, timedelta

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import src.main as main


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _forge_hs256(payload: dict, secret: str) -> str:
    """Ghép TAY một token HS256 (không qua PyJWT.encode, vốn tự chặn PEM làm secret).

    Mô phỏng đúng hành vi kẻ tấn công: lấy public PEM đã biết làm HMAC secret,
    tự tính chữ ký. Phòng thủ thật sự nằm ở phía DECODE (algorithms=["ES256"]).
    """
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(payload).encode())}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def _keypair_pem():
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _exp(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


@pytest.mark.asyncio
async def test_valid_es256_gateway_token_accepted(monkeypatch):
    priv, pub = _keypair_pem()
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub)
    token = jwt.encode({"role": "gateway", "exp": _exp(2)}, priv, algorithm="ES256")
    result = await main.verify_gateway_jwt(_creds(token))
    assert result["role"] == "gateway"


@pytest.mark.asyncio
async def test_missing_public_key_fails_closed(monkeypatch):
    monkeypatch.delenv("GATEWAY_JWT_PUBLIC_KEY", raising=False)
    priv, _ = _keypair_pem()
    token = jwt.encode({"role": "gateway", "exp": _exp(2)}, priv, algorithm="ES256")
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(token))
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_token_from_wrong_key_rejected(monkeypatch):
    priv1, _ = _keypair_pem()
    _, pub2 = _keypair_pem()  # unrelated key pair
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub2)
    token = jwt.encode({"role": "gateway", "exp": _exp(2)}, priv1, algorithm="ES256")
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(token))
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_alg_confusion_hs256_rejected(monkeypatch):
    """Kẻ tấn công thử ký HS256 dùng chính public key làm secret — phải bị từ chối."""
    priv, pub = _keypair_pem()
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub)
    forged = _forge_hs256(
        {"role": "gateway", "exp": int(_exp(2).timestamp())}, pub
    )
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(forged))
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_expired_token_rejected(monkeypatch):
    priv, pub = _keypair_pem()
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub)
    token = jwt.encode(
        {"role": "gateway", "exp": datetime.now(timezone.utc) - timedelta(seconds=5)},
        priv,
        algorithm="ES256",
    )
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(token))
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_token_missing_exp_rejected(monkeypatch):
    """Token ký ES256 ĐÚNG khóa nhưng THIẾU exp phải bị từ chối (require exp).

    Gateway luôn ký kèm exp 2m; token không hạn dùng là dấu hiệu bất thường (vd private
    key rò rỉ bị lạm dụng tạo token vĩnh viễn). options={"require":["exp"]} chặn nó.
    """
    priv, pub = _keypair_pem()
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub)
    token = jwt.encode({"role": "gateway"}, priv, algorithm="ES256")  # KHÔNG có exp
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(token))
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_wrong_role_rejected(monkeypatch):
    priv, pub = _keypair_pem()
    monkeypatch.setenv("GATEWAY_JWT_PUBLIC_KEY", pub)
    token = jwt.encode({"role": "client", "exp": _exp(2)}, priv, algorithm="ES256")
    with pytest.raises(HTTPException) as ei:
        await main.verify_gateway_jwt(_creds(token))
    assert ei.value.status_code == 403
