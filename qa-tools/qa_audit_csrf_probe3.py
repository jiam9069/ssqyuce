import os, json
os.environ.setdefault("LOTT_BOOT_REFRESH_RULES","0")
os.environ["LOTT_AUTH_MODE"]="enforce"; os.environ.pop("LOTT_TOKEN",None)
os.environ.setdefault("LOTT_MASTER_KEY","0"*64)
from fastapi.testclient import TestClient
from lottery import config, db, api_app, auth
config.AUTH_MODE="enforce"; config.SESSION_COOKIE_SECURE=False; config.API_DOCS_ENABLED=False
db.close(); db.get_conn()
conn=db.get_conn()
conn.execute("UPDATE users SET password_hash=?, role='admin', status='active', is_primary_admin=1 WHERE status='bootstrap'",(auth.hash_password("AdminPass123"),))
conn.commit()
for u in ("alice","bob"):
    try: auth.create_user(u,u.capitalize()+"Pass123",role="user")
    except Exception: pass
draws=[{"issue":f"{2013+i//150}{i%150+1:03d}","date":"2020-01-01","reds":[3,7,11,15,22,30],"order":[3,7,11,15,22,30],"blue":(i%16)+1} for i in range(700)]
db.upsert_draws(draws)
C=TestClient(api_app.app); B=TestClient(api_app.app)
C.post("/api/auth/login",json={"username":"alice","password":"AlicePass123"})
B.post("/api/auth/login",json={"username":"bob","password":"BobPass123"})
csrf=C.get("/api/auth/me").json().get("csrf_token")

print("== predict WITHOUT CSRF (should 403) ==")
r=C.post("/api/predict",headers={"X-CSRF-Token":"WRONG"}); print("wrong csrf:",r.status_code,r.text[:120].replace("\n"," "))
r=C.post("/api/predict"); print("no csrf:",r.status_code,r.text[:120].replace("\n"," "))

print("\n== POST user/llm CSRF behavior (control) ==")
r=C.post("/api/user/llm/config",json={"model":"x"}); print("no csrf:",r.status_code,r.text[:120].replace("\n"," "))
r=C.post("/api/user/llm/config",json={"model":"x"},headers={"X-CSRF-Token":csrf}); print("good csrf:",r.status_code,r.text[:120].replace("\n"," "))

print("\n== n_tickets boundary ==")
r=C.post("/api/predict?n_tickets=0",headers={"X-CSRF-Token":csrf}); j=r.json(); print("n_tickets=0:",r.status_code,"actual",j.get("actual_tickets"),"shortfall",j.get("shortfall_reason"))
r=C.post("/api/predict?n_tickets=100000",headers={"X-CSRF-Token":csrf}); j=r.json(); print("n_tickets=100000:",r.status_code,"actual",j.get("actual_tickets"))

print("\n== predict GET huge n_tickets ==")
r=C.get("/api/predict?n_tickets=500000"); print("GET huge:",r.status_code,len(r.content),"bytes")

print("\n== logout then reuse cookie ==")
r=C.post("/api/auth/logout",headers={"X-CSRF-Token":csrf}); print("logout:",r.status_code)
print("me after logout:",C.get("/api/auth/me").json())

print("\n== forged cookie ==")
C2=TestClient(api_app.app); C2.cookies.set("__Host-ssq_sid","forged_session_abc")
print("forged cookie /api/predictions/last:",C2.get("/api/predictions/last").status_code)

print("\n== error leakage ==")
for p in ("/api/eval","/api/eval/recommendations","/api/patterns","/api/registry/status","/api/llm/audit/latest","/api/l2/distribution"):
    r=C.get(p)
    leak=any(s in r.text.lower() for s in ("traceback","/root/","sqlite","select ","file \""))
    print(p,r.status_code,"LEAK" if leak else "clean",r.text[:80].replace("\n"," "))
print("DONE")
