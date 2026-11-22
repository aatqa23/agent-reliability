# World and tools (implemented by A in env/tools.py)

Tools (all take the database first, return a string):
- lookup_customer(customer_id) -> name and address
- get_order(order_id) -> status, total, address, items, refunded_amount
- cancel_order(order_id, reason) -> sets status to "cancelled"
- modify_order_address(order_id, address) -> sets the order's address
- refund_order(order_id, amount) -> adds amount to refunded_amount

Policies (the agent is told these; tools enforce them and return ERROR otherwise):
1. Cancel or modify an order only while its status is "processing".
2. Refund only "delivered" orders, and the total refunded must not exceed the order total.
3. Refunds above 100 are not allowed (needs a human); the agent must refuse.
4. The agent must verify the order belongs to the customer who is asking.

Shared settings (everyone uses the same):
- Models: openai/gpt-oss-120b (large), openai/gpt-oss-20b (small)
- temperature 0.7, reasoning_effort="low", 5 trials per task (seeds 0-4)

Writing tasks (10 each, ids A01-A10 / B01-B10 / C01-C10):
- Start from env/base_db.json; assertions test the final database state.
- Include some tasks the agent must REFUSE (state must stay unchanged): assert the
  unchanged values and add "refusal": true to the task so the validator allows it.
- Include 2-3 step tasks (e.g. cancel one order and change the address of another).
- Spread your tasks over different orders so tasks can be chained later.
- Check by hand that each task is solvable, then run `python contract.py`.
