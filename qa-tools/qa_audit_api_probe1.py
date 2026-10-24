"""Black-box probe for ssqyuce API. Run under isolated_test.sh."""
import os, json, sys, time
os.environ.setdefault("LOTT_BOOT_REFRESH_RULES", "0")
os.environ["LOTT_AUTH_MODE"] = "enforce"
os.environ.pop("LOTT_TOKEN", None)
os.environ.setdefault("LOTT_MASTER_KEY", "0"*64)

from fastapi.testclient import TestClient
from lottery import config, db, api_app, auth

config.AUTH_MODE = "enforce"
config.SESSION_COOKIE_SECURE = False
config.API_DOCS_ENABLED = False
db.close()
db.get_conn()

# create admin + alice + bob
conn = db.get_conn()
conn.execute("UPDATE users SET password_hash=?, role='admin', status='active', is_primary_admin=1 WHERE status='bootstrap'",
             (auth.hash_password("AdminPass123"),))
conn.commit()
for u in ("alice","bob"):
    try: auth.create_user(u, u.capitalize()+"Pass123", role="user")
    except Exception as e: print("create_user", u, e)

c = TestClient(api_app.app)

def login(u,p):
    r = c.post("/api/auth/login", json={"username":u,"password":p})
    return r, r.cookies

def csrf(cookies):
    return cookies.get("csrf_token") or cookies.get("csrf") or ""

print("=== A. docs/openapi exposure (AUTH_MODE=enforce) ===")
for p in ("/docs","/openapi.json","/redoc"):
    r=c.get(p); print(p, r.status_code, len(r.text))

print("\n=== B. deny-by-default on data endpoints ===")
for p in ("/api/predictions/last","/api/draws/history","/api/capabilities","/api/info",
          "/api/eval","/api/patterns","/api/llm/arm","/api/registry/status",
          "/api/tasks","/api/llm/cost","/api/admin/users","/api/admin/audit",
          "/api/user/llm/config"):
    r=c.get(p); print(p, r.status_code)

print("\n=== C. static traversal / .env ===")
for p in ("/static/../.env","/static/..%2f.env","/static/../../lottery/config.py","/static/does-not-exist"):
    r=c.get(p); print(p, r.status_code, r.text[:60].replace("\n"," "))

print("\n=== D. login + auth flows ===")
ra, ca = login("alice","AlicePass123")
print("alice login", ra.status_code, "cookies:", list(ca.keys()))
print("alice /api/auth/me", c.get("/api/auth/me").status_code, c.get("/api/auth/me").json())

rb, cb = login("bob","BobPass123")
print("bob login", rb.status_code)

print("\n=== E. missing param / type errors on predict & portfolio ===")
tests = [
 ("GET","/api/portfolio/expected-value","params"),
 ("GET","/api/portfolio/distribution","params"),
 ("GET","/api/portfolio/blue-coverage","params"),
 ("GET","/api/portfolio/variance-pick","params"),
 ("GET","/api/multiscale/scan","params"),
 ("GET","/api/rigor/signal","params"),
 ("GET","/api/rigor/walk_forward","params"),
]
for m,p,_ in tests:
    r=c.get(p); print(p, "->", r.status_code, r.text[:100].replace("\n"," "))

print("\n=== F. superlong / negative / type-mismatch query params ===")
r=c.get("/api/portfolio/variance-pick", params={"n":"-5"}); print("variance-pick n=-5", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/portfolio/variance-pick", params={"n":"abc"}); print("variance-pick n=abc", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/portfolio/variance-pick", params={"n":"999999999999999999999"}); print("variance-pick n=huge", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/portfolio/distribution", params={"budget":"-100"}); print("distribution budget=-100", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/portfolio/distribution", params={"budget":"nan"}); print("distribution budget=nan", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/draws/history", params={"limit":"-1"}); print("draws history limit=-1", r.status_code, r.text[:120].replace("\n"," "))
r=c.get("/api/draws/history", params={"limit":"999999"}); print("draws history limit=999999", r.status_code, r.text[:100].replace("\n"," "))

print("\n=== G. tenant isolation: alice vs bob llm config ===")
# alice writes her config
hooks = {}
def authz(cookies):
    # need csrf header
    pass
# Inspect csrf cookie name
print("alice cookies:", dict(ca))
csrf_alice = ca.get("csrf_token") or ca.get("csrf")
r=c.get("/api/user/llm/config"); print("alice GET llm config", r.status_code, r.text[:200])

print("\n=== H. cross-tenant: alice tries to read/write bob's row via user_id ===")
ra2 = c.get("/api/user/llm/config", params={"user_id":"999"}); print("GET with user_id=999", ra2.status_code, ra2.text[:150])

print("\n=== I. error leakage: force exceptions ===")
for p in ("/api/tasks/notexist","/api/tasks/..%2f..%2fetc","/api/predictions/history?limit=x"):
    r=c.get(p); print(p, r.status_code, r.text[:150].replace("\n"," "))

print("\n=== J. idempotency / predict generation (POST) ===")
# find predict generation endpoint
print("predict-related routes:", [r.path for r in api_app.app.routes if "predict" in getattr(r,'path','')])

print("\nDONE")
