"""
Bump a user's personal plan to PROFESSIONAL via admin override.

Usage on the VM:
    docker compose cp bump_user.py backend:/app/bump_user.py
    docker compose exec backend python /app/bump_user.py jacobasuquo199@gmail.com

This uses the SubscriptionInDB.is_admin_override + admin_override_plan
fields, so it does NOT touch Stripe. The override is server-authoritative:
get_user_subscription() returns admin_override_plan as the effective plan.

Pass an email and (optionally) a plan name. Defaults to PROFESSIONAL.
Plan options: free, starter, professional, enterprise
"""
import asyncio
import sys
from datetime import datetime


async def bump(email: str, plan_name: str = "professional") -> None:
    from backend.db.mongodb.mongodb import MongoDB
    from backend.models.billing import SubscriptionInDB, SubscriptionPlan, SubscriptionStatus

    plan_name = plan_name.lower()
    try:
        plan = SubscriptionPlan(plan_name)
    except ValueError:
        print(f"❌  unknown plan '{plan_name}'. Options: free, starter, professional, enterprise")
        sys.exit(1)

    db = await MongoDB.get_database()
    users = db.users
    subs = db.subscriptions

    user = await users.find_one({"email": email})
    if not user:
        print(f"❌  no user with email {email}")
        sys.exit(1)

    user_id = str(user["_id"])
    print(f"✓  found user {email}  id={user_id}")

    now = datetime.utcnow()
    sub = await subs.find_one({"user_id": user_id})

    if sub:
        await subs.update_one(
            {"_id": sub["_id"]},
            {"$set": {
                "is_admin_override": True,
                "admin_override_plan": plan.value,
                "admin_override_expires": None,
                "status": SubscriptionStatus.ACTIVE.value,
                "updated_at": now,
            }},
        )
        print(f"✓  updated subscription  plan_override={plan.value}  status=active")
    else:
        # stripe_customer_id is required by the model but isn't used
        # for admin-override flows. Stripe real IDs start with "cus_",
        # so the "admin_override:" prefix is collision-safe.
        new_sub = SubscriptionInDB(
            user_id=user_id,
            stripe_customer_id=f"admin_override:{user_id}",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            is_admin_override=True,
            admin_override_plan=plan,
            admin_override_expires=None,
        )
        doc = new_sub.model_dump()
        doc["created_at"] = now
        doc["updated_at"] = now
        await subs.insert_one(doc)
        print(f"✓  created subscription  plan_override={plan.value}  status=active")

    # Re-check via the service layer to confirm what the app will see
    from backend.services.billing_service import get_user_subscription
    resolved = await get_user_subscription(user_id)
    print(f"✓  effective plan now reads as: {resolved.plan.value}  (active={resolved.is_active})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python bump_user.py <email> [plan]")
        sys.exit(1)
    email = sys.argv[1]
    plan = sys.argv[2] if len(sys.argv) > 2 else "professional"
    asyncio.run(bump(email, plan))
