"""Black-box probe 2: tenant isolation of predictions + predict flow."""
import os, json
os.environ.setdefault("LOTT_BOOT_REFRESH_RULES", "0")
os.environ["LOTT_AUTH_MODE"] = "enforce"
os.environ.pop("LOTT_TOKEN", None)
os.environ.setdefault("LOTT_MASTER_KEY", "0"*64)

from fastapi.testclient import TestClient
from lottery import config, db, api_app, auth

config.AUTH_MODE = "enforce"
config.SESSION_COOKIE_SECURE = False
config.API_DOCS_ENABLED = False
db.close(); db.get_conn()
conn = db.get_conn()
conn.execute("UPDATE users SET password_hash=?, role='admin', status='active', is_primary_admin=1 WHERE status='bootstrap'",
             (auth.hash_password("AdminPass123"),))
conn.commit()
for u in ("alice","bob"):
    try: auth.create_user(u, u.capitalize()+"Pass123", role="user")
    except Exception: pass

draws=[]
for i in range(700):
    draws.append({"issue": f"{2013+i//150}{i%150+1:03d}","date":"2020-01-01",
                  "reds":[3,7,11,15,22,30],"order":[3,7,11,15,22,30],"blue":(i%16)+1})
db.upsert_draws(draws)

A = TestClient(api_app.app)
B = TestClient(api_app.app)

def login(c,u,p):
    return c.post("/api/auth/login",json={"username":u,"password":p})

la=login(A,"alice","AlicePass123"); lb=login(B,"bob","BobPass123")
print("alice login", la.status_code, "bob login", lb.status_code)
me=A.get("/api/auth/me").json()
csrf_a = me.get("csrf_token")
alice_uid = me["user"]["user_id"]
print("alice uid", alice_uid, "csrf", csrf_a)

db.save_predictions("2023120", [{"reds":[1,2,3,4,5,6],"blue":7,"confidence":0.5,"method":"stat"}], user_id=alice_uid)
print("\nalice predictions/last:", json.dumps(A.get("/api/predictions/last").json(), ensure_ascii=False)[:220])

meb = B.get("/api/auth/me").json(); csrf_b = meb.get("csrf_token")
print("bob uid:", meb["user"]["user_id"])
print("bob predictions/last (should NOT see alice's):", json.dumps(B.get("/api/predictions/last").json(), ensure_ascii=False)[:320])

print("\n== bob POST /api/predict ==")
r = B.post("/api/predict", headers={"x-csrf-token": csrf_b})
print("bob predict:", r.status_code, r.text[:300].replace("\n"," "))

print("\n== alice POST /api/predict ==")
r = A.post("/api/predict", headers={"x-csrf-token": csrf_a})
print("alice predict:", r.status_code, r.text[:400].replace("\n"," "))

print("\n== bob predictions/last after alice predicted ==")
print(json.dumps(B.get("/api/predictions/last").json(), ensure_ascii=False)[:320])

print("\n== predict n_tickets validation ==")
r=A.post("/api/predict", params={"n_tickets":"-3"}, headers={"x-csrf-token": csrf_a})
print("n_tickets=-3:", r.status_code, r.text[:200].replace("\n"," "))
r=A.post("/api/predict", params={"n_tickets":"999999999"}, headers={"x-csrf-token": csrf_a})
print("n_tickets=huge:", r.status_code, r.text[:200].replace("\n"," "))

print("\n== CSRF: predict without csrf header ==")
r=A.post("/api/predict")
print("no csrf:", r.status_code, r.text[:150].replace("\n"," "))

print("\n== user_llm write cross-tenant ==")
payload={"base_url":"https://evil.example/v1","api_key":"sk-alicekey","model":"m","use_custom_key":True,"user_id": meb["user"]["user_id"]}
r=A.put("/api/user/llm/config", json=payload, headers={"x-csrf-token": csrf_a})
print("alice PUT user/llm with bob uid:", r.status_code, r.text[:250].replace("\n"," "))
print("bob GET llm config:", json.dumps(B.get("/api/user/llm/config").json(), ensure_ascii=False)[:280])

print("\n== admin endpoints as normal user (expect 403 if enforce) ==")
for p in ("/api/admin/users","/api/admin/audit","/api/llm/cost","/api/tasks"):
    r=A.get(p); print(p, r.status_code)

print("\n== delete-to-empty / re-predict idempotency ==")
n1 = db.get_conn().execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",(alice_uid,)).fetchone()[0]
A.post("/api/predict", params={"regenerate":"true"}, headers={"x-csrf-token": csrf_a})
n2 = db.get_conn().execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",(alice_uid,)).fetchone()[0]
print("alice pred count before/after regenerate:", n1, n2)

print("\nDONE")
