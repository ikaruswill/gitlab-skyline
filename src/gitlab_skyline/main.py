import argparse
import asyncio
import datetime
import json
import logging
import math
import operator
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import List, Tuple

import aiohttp
import requests
import solid2
from solid2 import (cube, import_stl, linear_extrude, polyhedron, rotate,
                    scad_render_to_file, scale, text, translate)

__author__ = "Will Ho"

logger = logging.getLogger(__name__)
GITLAB_MAX_EVENTS_PER_PAGE = 100
CACHE_EXPIRY_HOURS = 24


def _init_logger():
    log_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)


def get_dates_in_year(year: int) -> List[datetime.date]:
    """Get all dates in specified year"""
    start_date = datetime.date(year, 1, 1)
    end_date = min(datetime.date(year + 1, 1, 1), datetime.datetime.now(tz=datetime.timezone.utc).date())
    num_days = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=n) for n in range(num_days)]


def get_userid(username: str, gitlab_url) -> int:
    """Get GitLab User ID from Username via GitLab Users API"""
    path = f"/api/v4/users?username={username}"
    url = urllib.parse.urljoin(gitlab_url, path)
    response_json = requests.get(url, timeout=30).json()
    try:
        userid = response_json[0]["id"]
    except IndexError:
        raise ValueError(f"User @{username} does not exist")

    logger.info(f"User ID for @{username} is {userid}")
    return userid


def get_output_filename(url: str, username: str, year: int):
    domain = urllib.parse.urlparse(url).netloc
    return f"{domain}_{username}_{year}"


def put_contributions_cache(date_contributions: List[Tuple[datetime.date, int]], path: Path):
    date_contributions_map = {d.strftime("%Y-%m-%d"): c for d, c in sorted(date_contributions)}

    with open(path, 'w') as f:
        json.dump(date_contributions_map, f)


def get_contributions_cache(path: Path) -> List[Tuple[datetime.date, int]]:
    if not path.exists():
        raise ValueError(f"Cache does not exist ({path})")
    if not path.is_file():
        raise ValueError(f"Cache is not a file ({path})")
    cache_expiry = datetime.timedelta(hours=CACHE_EXPIRY_HOURS)
    cache_modified_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    current_time = datetime.datetime.now()
    cache_age = current_time - cache_modified_time
    if cache_age >= cache_expiry:
        raise ValueError(f"Cache has expired ({cache_age.seconds / 3600} hours > {CACHE_EXPIRY_HOURS} hours) ({path})")

    with open(path) as f:
        try:
            date_contributions_map = json.load(f)
        except ValueError as e:
            raise ValueError(f"Cache is invalid JSON ({path}): {e}")

    logger.debug(f"Cache is valid ({path})")
    date_contributions = [
        (datetime.datetime.strptime(d, "%Y-%m-%d").date(), c) for d, c in date_contributions_map.items()
    ]
    date_contributions.sort()
    return date_contributions


def get_contributions(
    userid: int, dates: List[datetime.date], gitlab_url: str, token: str, concurrency: int
) -> List[Tuple[datetime.date, int]]:
    """Get contributions for User ID ordered by date"""
    date_contributions = []
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[
                get_contributions_for_date(
                    semaphore=semaphore,
                    gitlab_url=gitlab_url,
                    userid=userid,
                    token=token,
                    date=date,
                    date_contributions=date_contributions,
                )
                for date in dates
            ]
        )
    )
    loop.close()
    date_contributions.sort()
    return date_contributions


async def get_contributions_for_date(
    semaphore: asyncio.Semaphore,
    gitlab_url: str,
    userid: int,
    token: str,
    date: datetime.date,
    date_contributions: List[Tuple[datetime.date, int]],
):
    """Get contributions (async) for User ID on a specific date using GitLab events API"""
    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token

    async with aiohttp.ClientSession(raise_for_status=True, headers=headers) as client:
        try:
            path = f"/api/v4/users/{userid}/events"
            url = urllib.parse.urljoin(gitlab_url, path)
            after = (date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            before = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            params = {
                "after": after,
                "before": before,
                "page": 1,
                "per_page": GITLAB_MAX_EVENTS_PER_PAGE,
            }

            contributions_for_date = 0
            contribution_count_in_page = None
            while contribution_count_in_page is None or contribution_count_in_page == GITLAB_MAX_EVENTS_PER_PAGE:
                async with semaphore, client.get(url, params=params) as response:
                    logger.debug(f"GET: {url}")
                    json = await response.json()
                    logger.debug(f"RESPONSE: {json}")
                contribution_count_in_page = len(json)
                contributions_for_date += contribution_count_in_page
                params["page"] += 1

            date_contributions.append((date, contributions_for_date))
            logger.info(f"Contributions for {date}: {contributions_for_date}")

        except Exception as err:
            logger.error(f"Exception occured: {err}")
            pass


def get_first_contribution_index(date_contributions: List[Tuple[datetime.date, int]]) -> int:
    """Find the date and list index of the first contribution"""
    for index, (_, contribution_count) in enumerate(date_contributions):
        if contribution_count > 0:
            return index
    logger.warning("Unable to determine first contribution index: No contributions in range")
    return 0


def pad_date_contributions_weekdays(date_contributions: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Ensure that data starts with a Sunday and ends with a Saturday"""
    first_date = date_contributions[0][0]
    last_date = date_contributions[-1][0]
    logger.debug(f"First date: {first_date.strftime('%Y-%m-%d')}")
    logger.debug(f"Last date: {last_date.strftime('%Y-%m-%d')}")
    # Sun = 0, Mon = 1, Tue = 2, Wed = 3, Thu = 4, Fri = 5, Sat = 6
    pad_left_days = first_date.isoweekday() % 7
    # Sun = 6, Mon = 5, Tue = 4, Wed = 3, Thu = 2, Fri = 1, Sat = 0
    pad_right_days = 6 - last_date.isoweekday() % 7
    logger.debug(f"Left weekdays to pad: {pad_left_days}")
    logger.debug(f"Right weekdays to pad: {pad_right_days}")
    left_padding = [(first_date.strftime('%Y-%m-%d'), 0)] * pad_left_days
    right_padding = [(last_date.strftime('%Y-%m-%d'), 0)] * pad_right_days
    logger.debug(f"Left padding: {left_padding}")
    logger.debug(f"Right padding: {right_padding}")
    return left_padding + date_contributions + right_padding


def percentile(values, percent, key=lambda x: x):
    target_idx = (len(values) - 1) * percent
    next_idx = math.floor(target_idx)
    previous_idx = math.ceil(target_idx)
    if next_idx == previous_idx:
        return key(values[int(target_idx)])
    d0 = key(values[int(next_idx)]) * (previous_idx - target_idx)
    d1 = key(values[int(previous_idx)]) * (target_idx - next_idx)
    return d0 + d1


def cap_outliers(date_contributions: List[Tuple[str, int]], threshold_percentile: float) -> List[Tuple[str, int]]:
    _, contribution_counts = zip(*date_contributions)
    non_zero_sorted_contribution_counts = sorted([count for count in contribution_counts if count > 0])
    outlier_threshold = percentile(non_zero_sorted_contribution_counts, percent=threshold_percentile / 100)
    outliers = [
        (i, (date, outlier_threshold))
        for i, (date, count) in enumerate(date_contributions)
        if count > outlier_threshold
    ]

    result = date_contributions.copy()
    for i, date_contribution in outliers:
        result[i] = date_contribution

    logger.info(f"{len(outliers)} outliers capped to threshold value: {outlier_threshold}")
    logger.debug(f"Outliers: {outliers}")
    return result


def get_bar_heights(contribution_counts: List[int], max_height: float):
    max_count = max(contribution_counts)
    return [count / max_count * max_height for count in contribution_counts]


def generate_skyline_model(
    contribution_counts: List[int],
    username: str,
    year: int,
    logo_path: Path,
    logo_scale: float,
    logo_y: float,
    logo_text_margin: float,
    handle_x: float,
    engrave_depth: float,
) -> solid2.union:
    """Generate SCAD model of contributions"""
    if len(contribution_counts) % 7 > 0:
        msg = "Number of contributions is not perfectly divisible by 7, check that padding is applied correctly"
        raise ValueError(msg)

    base_length_warn_threshold = 100

    # Parameters
    base_top_offset = 3.5
    base_width = 30
    base_height = 10
    bar_base_dimension = 2.5
    bar_max_height = 20
    bar_l_margin = 2.5
    bar_w_margin = 2.5

    # Coupled text parameters: Any change requires tuning all visually
    text_size = 5
    font = "Helvetica"
    year_width_est = 14
    text_y = 2.5

    # Derived parameters
    num_columns = len(contribution_counts) / 7
    base_length = num_columns * bar_base_dimension + 2 * bar_l_margin + 2 * base_top_offset
    base_face_angle = math.degrees(math.atan(base_height / base_top_offset))

    if base_length < base_length_warn_threshold:
        logger.warning(
            f"Base length is less than {base_length_warn_threshold}mm, model may have issues."
            "Check and adjust SCAD file."
        )

    # Build base
    base_points = [
        [0, 0, 0],
        [base_length, 0, 0],
        [base_length, base_width, 0],
        [0, base_width, 0],
        [base_top_offset, base_top_offset, base_height],
        [base_length - base_top_offset, base_top_offset, base_height],
        [base_length - base_top_offset, base_width - base_top_offset, base_height],
        [base_top_offset, base_width - base_top_offset, base_height],
    ]

    base_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 1, 0],  # front
        [7, 6, 5, 4],  # top
        [5, 6, 2, 1],  # right
        [6, 7, 3, 2],  # back
        [7, 4, 0, 3],  # left
    ]

    base_scad = polyhedron(points=base_points, faces=base_faces)

    year_scad = rotate([base_face_angle, 0, 0])(
        translate([base_length - logo_text_margin - year_width_est, text_y, -engrave_depth])(
            linear_extrude(height=2)(text(text=str(year), size=text_size, font=font))
        )
    )

    user_scad = rotate([base_face_angle, 0, 0])(
        translate([handle_x, text_y, -engrave_depth])(
            linear_extrude(height=2)(text(text="@" + username, size=text_size, font=font))
        )
    )

    logo_scad = rotate([base_face_angle, 0, 0])(
        translate([logo_text_margin, logo_y, -engrave_depth])(
            linear_extrude(height=2)(scale([logo_scale, logo_scale, logo_scale])(import_stl(str(logo_path))))
        )
    )

    # Build bars
    bars = None
    bar_heights = get_bar_heights(contribution_counts=contribution_counts, max_height=bar_max_height)

    week_number = 0
    last_weekday = 6  # Saturday
    for i, bar_height in enumerate(bar_heights):
        day_number = i % 7
        bar_x_factor = week_number
        bar_y_factor = 6 - day_number

        bar = translate(
            [
                base_top_offset + bar_l_margin + bar_x_factor * bar_base_dimension,
                base_top_offset + bar_w_margin + bar_y_factor * bar_base_dimension,
                base_height,
            ]
        )(
            cube(
                [
                    bar_base_dimension,
                    bar_base_dimension,
                    bar_height,
                ]
            )
        )

        if bars is None:
            bars = bar
        else:
            bars += bar

        if day_number == last_weekday:
            week_number += 1

    scad_skyline_object = base_scad - logo_scad - user_scad - year_scad

    if bars is not None:
        scad_skyline_object += bars

    return scad_skyline_object


def render_scad(model: solid2.union, path: Path):
    scad_render_to_file(model, filename=path.name, out_dir=str(path.parent))
    logger.info(f"Generated SCAD file: {path}")


def render_stl(scad_path: Path, path: Path):
    try:
        subprocess.run(
            ["openscad", "-o", str(path), str(scad_path)],
            capture_output=True,
        )
        logger.info(f"Generated STL file: {path}")
    except FileNotFoundError:
        logger.error("'openscad' binary not found, is OpenSCAD installed?")
        logger.error(
            "Unable to generate STL file, "
            "you may export the STL manually by opening the generated SCAD file in OpenSCAD"
        )


def fix_url(url: str) -> str:
    if not url.startswith("https://") and not url.startswith("http://"):
        logger.warning("Scheme not provided in url argument, assuming 'https'...")
        return f"https://{url}"
    return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gitlab-skyline",
        description="Create OpenSCAD [and STL] models from GitLab contributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("username", type=str, help="GitLab username (without @)")
    parser.add_argument(
        "year",
        type=int,
        help="Year of contributions to fetch",
        default=datetime.datetime.now(tz=datetime.timezone.utc).year,
        nargs="?",
    )
    parser.add_argument("-t", "--truncate", action="store_true", help="Truncate dates before first contribution")
    parser.add_argument("-o", "--output", type=Path, help="Output path", default=Path.cwd().relative_to(Path.cwd()))
    parser.add_argument("--stl", action="store_true", help="Export an STL file as well (Requires openscad binary)")
    parser.add_argument("--url", type=str, help="GitLab URL", default="https://gitlab.com")
    parser.add_argument("--token", type=str, help="Personal access token", default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Max concurrent requests to GitLab",
        default=2,
    )
    parser.add_argument(
        "--logo",
        type=Path,
        help="Path to SVG of logo to be engraved on the front face",
        default=Path(__file__).parent.absolute() / "gitlab.svg",
    )
    parser.add_argument("--logo-scale", type=float, help="Logo scale factor", default=0.09)
    parser.add_argument("--logo-y", type=float, help="Logo y-offset from bottom (mm)", default=1.25)
    parser.add_argument("--logo-margin", type=float, help="Logo and text x margin from sides (mm)", default=10)
    parser.add_argument("--handle-x", type=float, help="Username handle x offset from left (mm)", default=35)
    parser.add_argument("--engrave-depth", type=float, help="Logo and text engrave depth (mm)", default=0.4)
    parser.add_argument(
        "--cap-pct", type=float, help="Percentile of non-zero contributions to cap outliers to (%)", default=95
    )
    parser.add_argument("--loglevel", type=str, help="Log level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()

    _init_logger()
    logger.setLevel(args.loglevel.upper())

    args.url = fix_url(args.url)
    output_filename = get_output_filename(url=args.url, username=args.username, year=args.year)
    cache_path = args.output / f"{output_filename}.json"

    logger.debug(f"Checking for cached results at {cache_path}...")
    try:
        date_contributions = get_contributions_cache(path=cache_path)
        logger.info("Using cached results, skipping fetch...")
    except ValueError as e:
        logger.debug(f"No valid cached results found at {cache_path}")
        dates = get_dates_in_year(year=args.year)
        userid = get_userid(username=args.username, gitlab_url=args.url)

        logger.info("Fetching contributions from GitLab...")
        date_contributions = get_contributions(
            userid=userid, dates=dates, gitlab_url=args.url, token=args.token, concurrency=args.concurrency
        )
        logger.debug(f"Caching contributions to {cache_path}...")
        put_contributions_cache(date_contributions=date_contributions, path=cache_path)

    # TODO: Validate contributions
    logger.info(f"Capping outliers to the {args.cap_pct} percentile value...")
    date_contributions = cap_outliers(date_contributions=date_contributions, threshold_percentile=args.cap_pct)

    if args.truncate:
        logger.info("Truncating dates before first contribution")
        first_contribution_index = get_first_contribution_index(date_contributions=date_contributions)
        date_contributions = date_contributions[first_contribution_index:]

    date_contributions = pad_date_contributions_weekdays(date_contributions=date_contributions)
    logger.debug(f"Padded contributions: {date_contributions}")

    _, contribution_counts = list(zip(*date_contributions))
    logger.debug(f"Contribution counts {contribution_counts}")

    logger.info("Generating model...")
    if args.logo:
        logger.info(f"Using user-defined logo from: {args.logo}")
    if args.logo:
        logger.info(f"Using logo scale factor of: {args.logo_scale}")
    model = generate_skyline_model(
        contribution_counts=contribution_counts,
        username=args.username,
        year=args.year,
        logo_path=args.logo,
        logo_scale=args.logo_scale,
        logo_y=args.logo_y,
        logo_text_margin=args.logo_margin,
        handle_x=args.handle_x,
        engrave_depth=args.engrave_depth,
    )

    logger.info("Rendering models to file...")
    scad_path = args.output / f"{output_filename}.scad"
    stl_path = args.output / f"{output_filename}.stl"
    render_scad(model=model, path=scad_path)
    if args.stl:
        render_stl(scad_path=scad_path, path=stl_path)


if __name__ == "__main__":
    main()
