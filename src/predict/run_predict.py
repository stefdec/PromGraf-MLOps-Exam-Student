"""Generate mixed successful and failing traffic for the prediction API."""

import os
import random
import signal
import time

import requests


PREDICT_URL = os.getenv("PREDICT_URL", "http://bike-api:8080/predict")
REQUEST_INTERVAL_SECONDS = float(os.getenv("REQUEST_INTERVAL_SECONDS", "0.5"))
ERROR_RATE = float(os.getenv("ERROR_RATE", "0.2"))
REQUEST_COUNT = int(os.getenv("REQUEST_COUNT", "0"))  # 0 means run forever
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))

running = True


def stop(_signum, _frame):
    """Stop cleanly when the container receives SIGINT or SIGTERM."""
    global running
    running = False


def valid_payload():
    """Return a realistic, randomly generated bike-sharing request."""
    return {
        "temp": round(random.uniform(0.05, 0.95), 4),
        "atemp": round(random.uniform(0.05, 0.95), 4),
        "hum": round(random.uniform(0.2, 1.0), 4),
        "windspeed": round(random.uniform(0.0, 0.65), 4),
        "mnth": random.randint(1, 12),
        "hr": random.randint(0, 23),
        "weekday": random.randint(0, 6),
        "season": random.randint(1, 4),
        "holiday": random.randint(0, 1),
        "workingday": random.randint(0, 1),
        "weathersit": random.randint(1, 4),
    }


def send_request(session):
    """Send either a valid prediction or a request designed to fail."""
    if random.random() >= ERROR_RATE:
        kind = "valid"
        response = session.post(
            PREDICT_URL,
            json=valid_payload(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    else:
        kind = random.choice(("missing-field", "invalid-type", "wrong-method"))
        payload = valid_payload()

        if kind == "missing-field":
            payload.pop("temp")
            response = session.post(
                PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        elif kind == "invalid-type":
            payload["hr"] = "not-an-hour"
            response = session.post(
                PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        else:
            response = session.get(PREDICT_URL, timeout=REQUEST_TIMEOUT_SECONDS)

    return kind, response


def main():
    if not 0.0 <= ERROR_RATE <= 1.0:
        raise ValueError("ERROR_RATE must be between 0 and 1")
    if REQUEST_INTERVAL_SECONDS < 0:
        raise ValueError("REQUEST_INTERVAL_SECONDS cannot be negative")
    if REQUEST_COUNT < 0:
        raise ValueError("REQUEST_COUNT cannot be negative")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sent = 0
    successes = 0
    failures = 0
    print(
        f"Generating traffic on {PREDICT_URL} "
        f"(error rate: {ERROR_RATE:.0%}, count: {REQUEST_COUNT or 'unlimited'})",
        flush=True,
    )

    with requests.Session() as session:
        while running and (REQUEST_COUNT == 0 or sent < REQUEST_COUNT):
            sent += 1
            try:
                kind, response = send_request(session)
                if response.ok:
                    successes += 1
                else:
                    failures += 1
                print(
                    f"request={sent} kind={kind} status={response.status_code} "
                    f"successes={successes} errors={failures}",
                    flush=True,
                )
            except requests.RequestException as exc:
                failures += 1
                print(
                    f"request={sent} kind=connection-error error={exc} "
                    f"successes={successes} errors={failures}",
                    flush=True,
                )

            if running and (REQUEST_COUNT == 0 or sent < REQUEST_COUNT):
                time.sleep(REQUEST_INTERVAL_SECONDS)

    print(
        f"Traffic generator stopped: sent={sent}, "
        f"successes={successes}, errors={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
