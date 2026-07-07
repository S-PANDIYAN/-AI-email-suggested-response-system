"""
Synthetic customer-support dataset generator.

Why synthetic?
--------------
Real support inboxes are PII-laden and can't be shipped in a repo. For an
evaluation-focused challenge we need data that is (a) redistributable,
(b) deterministic (same every run -> reproducible scores), and (c) varied
enough that semantic retrieval is non-trivial. Templated generation with a
seeded RNG gives all three.

What "realistic" means here
---------------------------
Each category has multiple *intent templates* (e.g. shipping has "where is my
order", "wrong item", "damaged in transit"), each with slot variety (order
ids, products, cities). Replies mirror real support style: acknowledge ->
empathise -> action -> next step. This variety is what makes RAG useful: a
good retriever must find the *intent* match, not just keyword overlap.

Schema (per row):  id, customer_email, support_reply, category
Output          :  dataset/data/emails.csv

Complexity: O(n) in number of pairs; trivial memory.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from config.settings import load_settings
from utils.io import ensure_dir
from utils.logger import get_logger

log = get_logger(__name__)

# ---- slot vocabularies (shared across templates) ------------------------
PRODUCTS = [
    "wireless headphones", "mechanical keyboard", "standing desk",
    "office chair", "4K monitor", "USB-C hub", "laptop stand",
    "webcam", "desk lamp", "ergonomic mouse",
]
CITIES = ["Austin", "Denver", "Seattle", "Boston", "Chicago", "Miami"]
NAMES = ["Alex", "Priya", "Jordan", "Sam", "Lena", "Marco", "Nina", "Tom"]

# Each category -> list of (email_template, reply_template) intent variants.
# {oid}=order id, {p}=product, {city}=city, {name}=agent name, {days}=int.
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Shipping": [
        ("My order {oid} for the {p} was supposed to arrive {days} days ago "
         "but the tracking hasn't updated. Where is it?",
         "Hi, thanks for reaching out. I'm sorry your {p} (order {oid}) is "
         "delayed. I've checked with the carrier — it's currently in transit "
         "and expected within {days} business days. Here is your live tracking "
         "link. I'll personally monitor it and update you if anything changes. "
         "— {name}"),
        ("I received the wrong item for order {oid}. I ordered a {p} but got "
         "something else entirely.",
         "Hi, I'm really sorry about the mix-up on order {oid}. I've created a "
         "prepaid return label for the incorrect item and shipped the correct "
         "{p} today at no cost to you. It should arrive in {days} business "
         "days. Apologies again for the inconvenience. — {name}"),
    ],
    "Refund": [
        ("I returned my {p} two weeks ago (order {oid}) and still haven't seen "
         "a refund. Please help.",
         "Hi, thank you for your patience. I can confirm we received your "
         "returned {p} for order {oid}. I've just processed your full refund — "
         "it will appear on your original payment method within {days} business "
         "days. You'll get an email confirmation shortly. — {name}"),
        ("The {p} didn't meet my expectations. I'd like a refund for order "
         "{oid}.",
         "Hi, I'm sorry the {p} wasn't the right fit. You're well within our "
         "30-day window, so I've approved your refund for order {oid}. Once you "
         "drop off the item using the attached label, the refund of the full "
         "amount posts within {days} business days. — {name}"),
    ],
    "Replacement": [
        ("My {p} from order {oid} arrived damaged — the screen is cracked.",
         "Hi, I'm so sorry your {p} arrived damaged. That's not the experience "
         "we want. I've shipped a replacement {p} today via express (order "
         "{oid}), arriving in {days} business days, and included a return label "
         "for the damaged unit. No further action needed on your end. — {name}"),
        ("The {p} I bought stopped working after a week. Can I get a "
         "replacement? Order {oid}.",
         "Hi, sorry to hear the {p} failed so quickly — it's covered under "
         "warranty. I've queued a replacement for order {oid} shipping today, "
         "ETA {days} business days. Please recycle the faulty unit; no need to "
         "return it. Let me know if the new one gives any trouble. — {name}"),
    ],
    "Billing": [
        ("I was charged twice for order {oid}. Can you fix this?",
         "Hi, thanks for flagging this. You're right — I can see a duplicate "
         "charge on order {oid}. I've reversed the extra charge immediately; it "
         "will drop off your statement within {days} business days. Apologies "
         "for the worry. — {name}"),
        ("There's a charge on my card I don't recognise, possibly related to "
         "my {p} purchase.",
         "Hi, I understand how concerning an unexpected charge is. I looked "
         "into it: the charge corresponds to your {p} order plus applicable "
         "tax. I've emailed you the itemised receipt. If it still looks wrong, "
         "reply here and I'll escalate to our billing team. — {name}"),
    ],
    "Subscription": [
        ("I want to upgrade my subscription plan but the page keeps erroring.",
         "Hi, sorry the upgrade page is misbehaving. I've applied the upgrade "
         "to your account manually — you now have the higher tier effective "
         "today, prorated for this cycle. You should see the new features "
         "within a few minutes. — {name}"),
        ("Why did my subscription price go up this month?",
         "Hi, great question. Your introductory promotional rate ended this "
         "cycle, so the plan returned to its standard price. I've applied a "
         "one-time loyalty credit to soften the change and emailed you the "
         "breakdown. Let me know if you'd like to review other plans. — {name}"),
    ],
    "Technical Support": [
        ("The {p} won't connect to my laptop over Bluetooth no matter what I "
         "try.",
         "Hi, let's get your {p} connected. Please (1) forget the device in "
         "Bluetooth settings, (2) hold the pairing button for 5 seconds until "
         "the light blinks, then (3) re-pair. This resolves it in most cases. "
         "If it still fails, reply and I'll arrange a replacement. — {name}"),
        ("Your app keeps crashing on startup after the latest update.",
         "Hi, sorry about the crashes. Our team identified a bug in the latest "
         "build. Please update to version 4.2.1 (now live) or reinstall — that "
         "clears the corrupted cache causing it. I've credited your account for "
         "the disruption. — {name}"),
    ],
    "Password Reset": [
        ("I can't log in and the password reset email never arrives.",
         "Hi, sorry for the login trouble. I've manually triggered a reset link "
         "to your registered email — please also check spam. It expires in 30 "
         "minutes. If it still doesn't arrive, confirm your account email and "
         "I'll verify it's correct on our side. — {name}"),
        ("I forgot my password and the reset link says it expired.",
         "Hi, no problem — reset links expire after 30 minutes for security. "
         "I've sent a fresh link that's valid now. Click it, choose a new "
         "password, and you'll be back in. Let me know if you hit any snag. "
         "— {name}"),
    ],
    "Account Issues": [
        ("My account shows the wrong shipping address and I can't edit it.",
         "Hi, thanks for letting me know. I've unlocked and corrected the "
         "shipping address on your account — please double-check it now looks "
         "right. The edit restriction was caused by a pending order; it's "
         "cleared now. — {name}"),
        ("I think my account was accessed by someone else.",
         "Hi, I take this seriously. I've secured your account: forced a "
         "sign-out of all sessions and locked new logins pending a reset. "
         "Please use the reset link I just sent to set a new password. I've "
         "also enabled login alerts. — {name}"),
    ],
    "Cancellation": [
        ("Please cancel my order {oid} for the {p}, I ordered by mistake.",
         "Hi, done — I've cancelled order {oid} for the {p} before it shipped, "
         "so you won't be charged. Any pending authorisation will drop off "
         "within {days} business days. Thanks for catching it early. — {name}"),
        ("I'd like to cancel my subscription effective immediately.",
         "Hi, I've cancelled your subscription effective today; you won't be "
         "billed again. You'll keep access until the end of the current paid "
         "period. We'd love feedback on why you're leaving if you have a "
         "moment. — {name}"),
    ],
    "General Questions": [
        ("Do you ship internationally, and how long does it take to {city}?",
         "Hi, yes! We ship to most countries. To {city}, standard international "
         "delivery is typically {days} business days plus any customs "
         "clearance. You'll get full tracking at dispatch. Let me know if "
         "you'd like the express option. — {name}"),
        ("What's your warranty policy on the {p}?",
         "Hi, great question. The {p} comes with a 2-year limited warranty "
         "covering manufacturing defects. If anything fails in that window, we "
         "repair or replace it free. Keep your order confirmation as proof of "
         "purchase. — {name}"),
    ],
}


def build_dataset(n_pairs: int, seed: int) -> pd.DataFrame:
    """Generate ``n_pairs`` rows, round-robin across categories/intents."""
    rng = random.Random(seed)                       # local RNG -> deterministic
    categories = list(TEMPLATES.keys())
    rows: list[dict] = []

    i = 0
    while len(rows) < n_pairs:
        category = categories[i % len(categories)]
        variants = TEMPLATES[category]
        email_t, reply_t = variants[(i // len(categories)) % len(variants)]
        slots = dict(
            oid=f"#{rng.randint(10000, 99999)}",
            p=rng.choice(PRODUCTS),
            city=rng.choice(CITIES),
            name=rng.choice(NAMES),
            days=rng.randint(2, 7),
        )
        rows.append(
            {
                "id": len(rows),
                "customer_email": email_t.format(**slots),
                "support_reply": reply_t.format(**slots),
                "category": category,
            }
        )
        i += 1

    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings()
    df = build_dataset(settings.dataset["n_pairs"], settings.seed)
    out = settings.paths["dataset_csv"]
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    log.info("Wrote %d pairs across %d categories -> %s",
             len(df), df["category"].nunique(), out)


if __name__ == "__main__":
    main()
