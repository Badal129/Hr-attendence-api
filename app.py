
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

from bson import BSON
from bson.errors import InvalidDocument
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

# Load local .env without overriding real environment variables.
load_dotenv(override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

if not MONGO_URI or not MONGO_DB:
    raise RuntimeError("Set MONGO_URI and MONGO_DB environment variables")

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc

client = MongoClient(
    MONGO_URI,
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[MONGO_DB]
employees = db["employees"]
attendance = db["attendance_logs"]

app = FastAPI(title="HR Attendance & Analytics API", version="1.0.0")

PRESENT_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = PRESENT_STATUSES | {"ABSENT", "LEAVE"}


@app.on_event("startup")
def create_indexes():
    # Unique employee business identity.
    employees.create_index(
        [("emp_code", ASCENDING)],
        unique=True,
        name="employee_code_unique",
    )

    # A single employee can have only one attendance record per date.
    attendance.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="attendance_employee_date_unique",
    )

    # Support attendance date-range and status filtering.
    attendance.create_index(
        [("date", ASCENDING), ("status", ASCENDING), ("emp_code", ASCENDING)],
        name="attendance_date_status_employee",
    )

    # Support department joins and employee filtering.
    employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="employee_department_code",
    )


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")


class EmployeeCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emp_code: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    department: str = Field(min_length=1, max_length=100)
    shift_start: str = Field(pattern=r"^\d{2}:\d{2}$")
    shift_end: str = Field(pattern=r"^\d{2}:\d{2}$")
    joined_on: date
    created_at: int = Field(ge=0)


class PunchInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emp_code: str
    punched_at: int = Field(ge=0)
    status: str


class AttendanceFilter(BaseModel):
    pass


def epoch_ms_to_datetime(value: int) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(
            status_code=422,
            detail="Timestamp must be an integer in epoch milliseconds",
        )

    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).replace(
            microsecond=0
        )
    except (OverflowError, OSError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid timestamp")


def datetime_to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    value = value.astimezone(UTC).replace(microsecond=0)
    return int(value.timestamp() * 1000)


def parse_shift(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid shift time")

    if parsed.second or parsed.microsecond or parsed.tzinfo:
        raise HTTPException(
            status_code=422,
            detail="Shift time must use HH:MM format",
        )

    return parsed


def calculate_late_minutes(
    punch_in: datetime,
    shift_start: str,
    shift_end: str,
) -> int:
    local = punch_in.astimezone(IST)
    start = parse_shift(shift_start)
    end = parse_shift(shift_end)

    shift_date = local.date()

    # An early-morning punch belongs to the previous day's
    # overnight shift when shift_end <= shift_start.
    if end <= start and local.time().replace(tzinfo=None) < end:
        shift_date -= timedelta(days=1)

    scheduled = datetime.combine(shift_date, start, tzinfo=IST)
    delta = local - scheduled

    # Grace period is strictly more than ten minutes.
    if delta.total_seconds() <= 600:
        return 0

    return max(0, int(delta.total_seconds() // 60))


def calculate_work_hours(
    punch_in: datetime,
    punch_out: datetime,
) -> float:
    seconds = int((punch_out - punch_in).total_seconds())

    value = Decimal(seconds) / Decimal(3600)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calculate_overtime_minutes(
    punch_out: datetime,
    shift_start: str,
    shift_end: str,
) -> int:
    local = punch_out.astimezone(IST)
    start = parse_shift(shift_start)
    end = parse_shift(shift_end)

    shift_date = local.date()

    if end <= start and local.time().replace(tzinfo=None) < end:
        shift_date -= timedelta(days=1)

    shift_end_date = shift_date
    if end <= start:
        shift_end_date += timedelta(days=1)

    scheduled_end = datetime.combine(shift_end_date, end, tzinfo=IST)
    minutes = int((local - scheduled_end).total_seconds() // 60)

    return max(0, minutes) if minutes >= 30 else 0


def serialize_attendance(doc: dict) -> dict:
    result = {
        key: value
        for key, value in doc.items()
        if key != "_id"
    }

    for key in ("punch_in", "punch_out"):
        result[key] = datetime_to_epoch_ms(result.get(key))

    result["history"] = result.get("history") or []
    result["half_day"] = result.get("half_day", False)
    return result


def validate_month(month: str) -> tuple[date, date]:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=422, detail="month must be YYYY-MM")

    try:
        first = date.fromisoformat(month + "-01")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid month")

    if first.month == 12:
        next_month = date(first.year + 1, 1, 1)
    else:
        next_month = date(first.year, first.month + 1, 1)

    return first, next_month


@app.post("/employees", status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate):
    parse_shift(payload.shift_start)
    parse_shift(payload.shift_end)

    if not payload.email.strip():
        raise HTTPException(status_code=422, detail="Email is required")

    doc = payload.model_dump()
    doc["emp_code"] = doc["emp_code"].strip()
    doc["name"] = doc["name"].strip()
    doc["email"] = doc["email"].strip()
    doc["department"] = doc["department"].strip()
    doc["created_at"] = epoch_ms_to_datetime(doc["created_at"])

    if doc["joined_on"] > doc["created_at"].astimezone(IST).date():
        raise HTTPException(
            status_code=422,
            detail="joined_on cannot be after created_at",
        )

    try:
        employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Employee already exists")

    doc.pop("_id", None)
    doc["created_at"] = datetime_to_epoch_ms(doc["created_at"])
    return doc


@app.get("/employees")
def list_employees(
    department: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filters = {}
    if department:
        filters["department"] = department

    total = employees.count_documents(filters)
    docs = list(
        employees.find(filters, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )

    for doc in docs:
        doc["created_at"] = datetime_to_epoch_ms(doc.get("created_at"))

    return {
        "items": docs,
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@app.post("/attendance/punch-in", status_code=status.HTTP_201_CREATED)
def punch_in(payload: PunchInRequest):
    if payload.status not in PRESENT_STATUSES:
        raise HTTPException(
            status_code=422,
            detail="status must be PRESENT, WFH or ON_DUTY",
        )

    employee = employees.find_one({"emp_code": payload.emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    instant = epoch_ms_to_datetime(payload.punched_at)
    local = instant.astimezone(IST)
    shift_start = parse_shift(employee["shift_start"])
    shift_end = parse_shift(employee["shift_end"])

    record_date = local.date()
    if (
        shift_end <= shift_start
        and local.time().replace(tzinfo=None) < shift_end
    ):
        record_date -= timedelta(days=1)

    doc = {
        "emp_code": payload.emp_code,
        "date": record_date.isoformat(),
        "status": payload.status,
        "punch_in": instant,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": calculate_late_minutes(
            instant, employee["shift_start"], employee["shift_end"]
        ),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        attendance.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="Attendance record already exists for this date",
        )

    return serialize_attendance(doc)


@app.get("/attendance")
def list_attendance(
    emp_code: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    record_status: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must be on or before end_date",
        )

    if record_status and record_status not in ALL_STATUSES:
        raise HTTPException(status_code=422, detail="Invalid status")

    filters = {}

    if emp_code:
        filters["emp_code"] = emp_code

    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date.isoformat()
        if end_date:
            date_filter["$lte"] = end_date.isoformat()
        filters["date"] = date_filter

    if record_status:
        filters["status"] = record_status

    total = attendance.count_documents(filters)

    docs = list(
        attendance.find(filters, {"_id": 0})
        .sort(
            [("date", DESCENDING), ("emp_code", ASCENDING)]
        )
        .skip((page - 1) * page_size)
        .limit(page_size)
    )

    return {
        "items": [serialize_attendance(doc) for doc in docs],
        "page": page,
        "page_size": page_size,
        "total": total,
    }