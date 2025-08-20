# mail_report.py
import os
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

def send_mail(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    sender: str,
    to_addrs: list[str],
    subject: str,
    body: str,
    attachments: list[str] = None,
    use_starttls: bool = True,
):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)

    for path in attachments or []:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            data = f.read()
        maintype = "application"
        subtype = "octet-stream"
        msg.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=p.name,
        )

    if use_starttls:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)

if __name__ == "__main__":
    # 從環境變數讀設定（由 GitHub Secrets 提供）
    SMTP_HOST = os.environ["SMTP_HOST"]
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ["SMTP_USER"]
    SMTP_PASS = os.environ["SMTP_PASS"]
    EMAIL_FROM = os.environ["EMAIL_FROM"]
    EMAIL_TO = [x.strip() for x in os.environ["EMAIL_TO"].split(",") if x.strip()]
    SUBJECT = os.environ.get("EMAIL_SUBJECT", "Daily Taiwan Stocks Report")
    BODY = os.environ.get("EMAIL_BODY", "附上今日自動產生的台股技術指標報告。")

    ATTACH = os.environ.get("EMAIL_ATTACHMENT", "ETF100_report.xlsx")
    SEND_TLS = os.environ.get("SMTP_STARTTLS", "true").lower() == "true"

    send_mail(
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        smtp_user=SMTP_USER,
        smtp_pass=SMTP_PASS,
        sender=EMAIL_FROM,
        to_addrs=EMAIL_TO,
        subject=SUBJECT,
        body=BODY,
        attachments=[ATTACH],
        use_starttls=SEND_TLS,
    )
    print("Email sent.")
