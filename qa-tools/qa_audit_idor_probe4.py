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
A=TestClient(api_app.app); B=TestClient(api_app.app)
A.post("/api/auth/login",json={"username":"alice","password":"AlicePass123"})
B.post("/api/auth/login",json={"username":"bob","password":"BobPass123"})
csrfA=A.get("/api/auth/me").json()["csrf_token"]; csrfB=B.get("/api/auth/me").json()["csrf_token"]

print("== cross-tenant LLM write: alice supplies bob's user_id ==")
bob_uid = B.get("/api/auth/me").json()["user"]["user_id"]
r=A.post("/api/user/llm/config", json={"model":"m","base_url":"https://evil/v1","api_key":"sk-alice","use_custom_key":True,"user_id":bob_uid}, headers={"X-CSRF-Token":csrfA})
print("alice POST user/llm with bob uid:", r.status_code, r.text[:200].replace("\n"," "))
print("bob view:", json.dumps(B.get("/api/user/llm/config").json(), ensure_ascii=False)[:200])

print("\n== admin user-management: normal user hits /api/admin/users ==")
print("alice /api/admin/users:", A.get("/api/admin/users").status_code)
print("alice POST create user:", A.post("/api/admin/users", json={"username":"x","password":"Xx12345678","role":"admin"}, headers={"X-CSRF-Token":csrfA}).status_code)

print("\n== IDOR: alice tries admin reset-password of bob ==")
print("alice reset bob pw:", A.post(f"/api/admin/users/{bob_uid}/reset-password", json={"password":"Hacked12345"}, headers={"X-CSRF-Token":csrfA}).status_code)

print("\n== tasks/{task_id} cross-tenant IDOR ==")
# create a task for bob then alice read it
conn=db.get_conn()
try:
    conn.execute("INSERT INTO tasks(user_id,task_id,kind,status,created_at) VALUES(?,?,?,?,?)",(bob_uid,"TASKB1","predict","done",1))
    conn.commit()
    print("bob task created")
except Exception as e:
    print("tasks schema:", [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()], e)
r=A.get("/api/tasks/TASKB1"); print("alice GET bob task:", r.status_code, r.text[:150].replace("\n"," "))

print("\n== concurrent predict duplicate submit (two rapid POST) ==")
r1=A.post("/api/predict?regenerate=true", headers={"X-CSRF-Token":csrfA})
r2=A.post("/api/predict?regenerate=true", headers={"X-CSRF-Token":csrfA})
print("r1",r1.status_code,"r2",r2.status_code)

print("\n== logout without csrf (should still logout?) ==")
print("logout no csrf:", A.post("/api/auth/logout").status_code)
n=db.get_conn().execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",(bob_uid,)).fetchone()[0]
print("alice predicted; issue count for alice:", db.get_conn().execute("SELECT COUNT(DISTINCT issue) FROM predictions WHERE user_id=16").fetchone()[0])
print("DONE")
