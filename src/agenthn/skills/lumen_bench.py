"""Synthetic expertise benchmark: a fictional product ("Lumen") the base can't know.

SOURCE = the product docs the agent studies. QUESTIONS = held-out customer
questions with accepted answer phrases (substring-matched after normalization).
Fictional + specific, so base ~0 and there's room for studying to help. ~24
questions so each is ~4% -> a smooth accuracy curve.
"""

SOURCE = """Lumen is a cloud analytics platform, founded in 2021 and headquartered in Denver.

Plans and pricing:
- Free: $0/month, up to 3 projects, 1 GB storage, 1 team member, community support only.
- Pro: $29/month, up to 25 projects, 50 GB storage, up to 10 team members, email
  support with a 24-hour response time, API access at 1,000 requests per hour, and a
  14-day free trial.
- Enterprise: $99/month per seat, unlimited projects, 1 TB storage, unlimited team
  members, SSO and audit logs, a dedicated account manager, phone support, a 4-hour
  support response time, and API access at 10,000 requests per hour.

Billing and policies:
- Refunds are available within 30 days of purchase.
- After cancellation, customer data is retained for 90 days, then permanently deleted.
- Annual billing includes 2 months free.
- Lumen is SOC 2 Type II certified.

Integrations: Slack, GitHub, Google Sheets, and Zapier. A Salesforce integration is
planned for Q3.

Features: real-time dashboards, scheduled email reports (daily, weekly, or monthly),
anomaly detection, CSV/PDF export, and custom themes. Anomaly detection is available
only on the Pro and Enterprise plans, and custom themes are Enterprise-only. The
maximum dashboard refresh rate is every 5 minutes on Pro and every 1 minute on Enterprise.
"""

# (question, [accepted answer phrases]) -- normalized substring match
QUESTIONS = [
    ("How much does the Pro plan cost per month?", ["29"]),
    ("What is the refund window?", ["30 day", "30day", "30"]),
    ("Which plan includes SSO?", ["enterprise"]),
    ("How many projects can a Free user create?", ["three", "3 project"]),
    ("What is the email support response time on the Pro plan?", ["24 hour", "24hour"]),
    ("How long is customer data kept after cancellation?", ["90 day", "90"]),
    ("What is the API rate limit on the Pro plan?", ["1000 req", "1000 per", "1000 request"]),
    ("On which plans is anomaly detection available?", ["enterprise"]),
    ("How long is the Pro free trial?", ["14 day", "14"]),
    ("How much storage does the Pro plan include?", ["50 gb", "50gb"]),
    ("Which integration is planned for Q3?", ["salesforce"]),
    ("What is the max dashboard refresh rate on Enterprise?", ["1 minute", "1minute", "every minute"]),
    ("How much does Enterprise cost per seat each month?", ["99"]),
    ("How much storage does the Enterprise plan include?", ["1 tb", "1tb"]),
    ("What is the support response time on Enterprise?", ["4 hour", "4hour"]),
    ("How much storage does the Free plan include?", ["1 gb", "1gb"]),
    ("How many team members can the Pro plan have?", ["10 team", "10 member", "10 user", "ten "]),
    ("What discount does annual billing give?", ["2 month", "two month"]),
    ("What security certification does Lumen have?", ["soc 2", "soc2"]),
    ("What is the API rate limit on Enterprise?", ["10000", "10 000"]),
    ("Which plan offers custom themes?", ["enterprise"]),
    ("Which plan includes phone support?", ["enterprise"]),
    ("Where is Lumen headquartered?", ["denver"]),
    ("How often can email reports be scheduled?", ["daily", "weekly", "monthly"]),
]


def _norm(s: str) -> str:
    return s.lower().replace("$", "").replace(",", "").replace("-", " ")


def is_correct(answer: str, accepts: list[str]) -> bool:
    a = _norm(answer)
    return any(_norm(x) in a for x in accepts)
