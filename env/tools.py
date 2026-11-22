"""Mock retail world: tools and policies. Every tool is fn(db, **kwargs) -> str and may mutate db.

Policy 4 (verify the order belongs to the requesting customer) is NOT enforced here on
purpose: tools do not know who is asking. The agent must compare get_order()'s
customer_id with the customer in the conversation. Tasks test this with refusal cases.
"""
import json

MAX_REFUND = 100.0

POLICY_TEXT = """You are a customer-service agent for an online electronics shop.
Follow these policies strictly:
1. Cancel or modify an order only while its status is "processing".
2. Refund only "delivered" orders, and the total refunded for an order must not exceed the order total.
3. Refunds above 100 are not allowed (they need a human). Refuse and explain.
4. Only act on an order that belongs to the customer who is asking. Check the order's customer_id
   against the customer in the conversation; if it does not match, refuse.
If a request breaks a policy, do not call the tool; explain briefly and finish."""


def _order(db: dict, order_id: str) -> dict:
    o = db["orders"].get(order_id)
    if o is None:
        raise ValueError(f"order '{order_id}' not found")
    return o


def lookup_customer(db: dict, customer_id: str) -> str:
    c = db["customers"].get(customer_id)
    if c is None:
        return f"ERROR: customer '{customer_id}' not found"
    return json.dumps({"customer_id": customer_id, **c})


def get_order(db: dict, order_id: str) -> str:
    try:
        o = _order(db, order_id)
    except ValueError as e:
        return f"ERROR: {e}"
    return json.dumps({"order_id": order_id, **o})


def cancel_order(db: dict, order_id: str, reason: str) -> str:
    try:
        o = _order(db, order_id)
    except ValueError as e:
        return f"ERROR: {e}"
    if o["status"] != "processing":
        return f"ERROR: order {order_id} is '{o['status']}' and cannot be cancelled (only 'processing')"
    o["status"] = "cancelled"
    return f"OK: order {order_id} cancelled (reason: {reason})"


def modify_order_address(db: dict, order_id: str, address: str) -> str:
    try:
        o = _order(db, order_id)
    except ValueError as e:
        return f"ERROR: {e}"
    if o["status"] != "processing":
        return f"ERROR: order {order_id} is '{o['status']}' and cannot be modified (only 'processing')"
    o["address"] = address
    return f"OK: order {order_id} address set to '{address}'"


def refund_order(db: dict, order_id: str, amount) -> str:
    try:
        o = _order(db, order_id)
        amount = float(amount)
    except ValueError as e:
        return f"ERROR: {e}"
    if o["status"] != "delivered":
        return f"ERROR: order {order_id} is '{o['status']}'; only 'delivered' orders can be refunded"
    if amount <= 0:
        return "ERROR: refund amount must be positive"
    if amount > MAX_REFUND:
        return f"ERROR: refunds above {MAX_REFUND:g} need a human agent"
    if o["refunded_amount"] + amount > o["total"] + 1e-9:
        return (f"ERROR: refund would exceed order total "
                f"({o['refunded_amount']:g} already refunded of {o['total']:g})")
    o["refunded_amount"] = round(o["refunded_amount"] + amount, 2)
    return f"OK: refunded {amount:g} on order {order_id} (total refunded {o['refunded_amount']:g})"


TOOLS = {
    "lookup_customer": lookup_customer,
    "get_order": get_order,
    "cancel_order": cancel_order,
    "modify_order_address": modify_order_address,
    "refund_order": refund_order,
}

# Shown to the agent in its prompt (name, description, argument names).
TOOL_SPECS = {
    "lookup_customer": ("Get a customer's name and address.", ["customer_id"]),
    "get_order": ("Get an order's customer_id, status, total, address, items and refunded_amount.", ["order_id"]),
    "cancel_order": ("Cancel an order.", ["order_id", "reason"]),
    "modify_order_address": ("Change an order's delivery address.", ["order_id", "address"]),
    "refund_order": ("Refund an amount (a number) on an order.", ["order_id", "amount"]),
}


def tool_docs() -> str:
    lines = []
    for name, (desc, args) in TOOL_SPECS.items():
        lines.append(f"- {name}({', '.join(args)}): {desc}")
    return "\n".join(lines)


def tool_schemas() -> list:
    """OpenAI-style function schemas for native tool calling (plus a 'finish' function)."""
    out = []
    for name, (desc, args) in TOOL_SPECS.items():
        props = {a: {"type": "number" if a == "amount" else "string"} for a in args}
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": args}}})
    out.append({"type": "function", "function": {
        "name": "finish",
        "description": "End the conversation with a final reply to the customer. "
                       "Call this when the request is done or when you must refuse.",
        "parameters": {"type": "object",
                       "properties": {"message": {"type": "string"}},
                       "required": ["message"]}}})
    return out
