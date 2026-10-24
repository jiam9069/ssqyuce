import os, json
os.environ.setdefault("LOTT_BOOT_REFRESH_RULES","0")
os.environ["LOTT_AUTH_MODE"]="enforce"; os.environ.pop("LOTT_TOKEN",None)
os.environ.setdefault("LOTT_MASTER_KEY","0"*64)
from fastapi.testclient import TestClient
from lottery import config, db, api_app, auth
config.AUTH_MODE="enforce"; config.SESSION_COOKIE_SECURE=False; config.API_DOCS_ENABLED=True
db.close(); db.get_conn()
conn=db.get_conn()
conn.execute("UPDATE users SET password_hash=?, role='admin', status='active', is_primary_admin=1 WHERE status='bootstrap'",(auth.hash_password("AdminPass123"),))
conn.commit()
try: auth.create_user("alice","AlicePass123",role="user")
except Exception: pass
C=TestClient(api_app.app)
C.post("/api/auth/login",json={"username":"alice","password":"AlicePass123"})

print("== docs when API_DOCS_ENABLED=True and enforce ==")
for p in ("/docs","/openapi.json"):
    r=C.get(p); print(p,r.status_code)

print("== force 500 to check stack leakage: bad type on endpoints ==")
# n_tickets non-int on predict GET -> FastAPI 422 (no leak). Try coverage_r huge, bet_mode weird
for q in ("bet_mode=__evil__","coverage_r=-999","coverage=notabool","use_llm=maybe"):
    r=C.get("/api/predict?"+q); print(q, r.status_code, r.text[:100].replace("\n"," "))

print("\n== portfolio endpoints param abuse ==")
for p,q in [("/api/portfolio/expected-value","limit=0"),("/api/portfolio/expected-value","limit=-1"),
            ("/api/portfolio/distribution","n_tickets=abc"),("/api/portfolio/blue-coverage","n_tickets=-5"),
            ("/api/multiscale/scan","window=abc"),("/api/rigor/walk_forward?","n=abc")]:
    r=C.get(p+"?"+q); print(p,q,"->",r.status_code,r.text[:90].replace("\n"," "))

print("\n== data endpoints with corrupt draw (invalid reds) ==")
# insert invalid draw
conn=db.get_conn()
try:
    conn.execute("INSERT INTO draws(issue,date,reds,blue) VALUES('2999999','2020-01-01','99,99,1,2,3,4',99)")
    conn.commit()
except Exception as e: print("insert err (schema):",e)
print("draws columns:", [r[1] for r in conn.execute("PRAGMA table_info(draws)").fetchall()])
print("DONE")
