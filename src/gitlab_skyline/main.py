import argparse
import asyncio
import datetime
import logging
import math
import operator
import subprocess
import sys
import urllib
from pathlib import Path
from typing import List, Tuple

import aiohttp
import requests
import solid2
from solid2 import cube, import_stl, linear_extrude, polyhedron, rotate, scad_render_to_file, scale, text, translate

__author__ = "Will Ho"

logger = logging.getLogger(__name__)


def _init_logger():
    log_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter("%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)


def get_userid(username: str, domain) -> int:
    """Get GitLab User ID from Username via GitLab Users API"""
    path = f"/api/v4/users?username={username}"
    url = urllib.parse.urljoin(domain, path)
    userid = requests.get(url, timeout=30).json()[0]["id"]
    logger.info(f"User ID for @{username} is {userid}")
    return userid


def get_contributions(
    userid: int, dates: List[datetime.date], domain: str, token: str, concurrency: int
) -> List[Tuple[str, int]]:
    """Get contributions for User ID"""
    date_contributions = []
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[
                get_contributions_for_date(
                    semaphore=semaphore,
                    domain=domain,
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
    return date_contributions


async def get_contributions_for_date(
    semaphore: asyncio.Semaphore,
    domain: str,
    userid: int,
    token: str,
    date: datetime.date,
    date_contributions: List[Tuple[str, int]],
):
    """Get contributions (async) for User ID on a specific date using GitLab events API"""
    logger.info(f"Getting contributions for: {date}")
    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token

    async with aiohttp.ClientSession(raise_for_status=True, headers=headers) as client:
        try:
            after = (date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            before = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            path = f"/api/v4/users/{userid}/events?after={after}&before={before}"
            url = urllib.parse.urljoin(domain, path)
            async with semaphore, client.get(url) as response:
                logger.debug(f"GET: {url}")
                json = await response.json()
                logger.debug(f"RESPONSE: {json}")
                date_contributions.append((date.strftime("%Y-%m-%d"), len(json)))

        except Exception as err:
            logger.error(f"Exception occured: {err}")
            pass


def get_dates_in_year(year: int) -> List[datetime.date]:
    """Get all dates in specified year"""
    start_date = datetime.date(year, 1, 1)
    end_date = min(datetime.date(year + 1, 1, 1), datetime.datetime.now(tz=datetime.timezone.utc).date())
    num_days = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=n) for n in range(num_days)]


def pad_contribution_counts_weekdays(
    ordered_contribution_counts: List[int], first_date: datetime.date, last_date: datetime.date
) -> List[int]:
    """Ensure that data starts with a Sunday and ends with a Saturday"""
    logger.debug(f"First date: {first_date.strftime('%Y-%m-%d')}")
    logger.debug(f"Last date: {last_date.strftime('%Y-%m-%d')}")
    # Sun = 0, Mon = 1, Tue = 2, Wed = 3, Thu = 4, Fri = 5, Sat = 6
    pad_left_days = first_date.isoweekday() % 7
    # Sun = 6, Mon = 5, Tue = 4, Wed = 3, Thu = 2, Fri = 1, Sat = 0
    pad_right_days = 6 - last_date.isoweekday() % 7
    logger.debug(f"Left weekdays to pad: {pad_left_days}")
    logger.debug(f"Right weekdays to pad: {pad_right_days}")
    left_padding = [0] * pad_left_days
    right_padding = [0] * pad_right_days
    logger.debug(f"Left padding: {left_padding}")
    logger.debug(f"Right padding: {right_padding}")
    return left_padding + ordered_contribution_counts + right_padding


def date_contributions_to_ordered_counts(date_contributions: List[Tuple[str, int]]) -> List[int]:
    """Sort by date and remove date from contributions"""
    sorted_date_contributions = sorted(date_contributions, key=operator.itemgetter(0))
    _, counts = list(zip(*sorted_date_contributions))
    return list(counts)


def generate_skyline_stl(contribution_counts: List[int], username: str, year: int) -> solid2.union:
    """Generate SCAD model of contributions"""
    if len(contribution_counts) % 7 > 0:
        msg = "Number of conributions is not perfectly divisible by 7, check that padding is applied correctly"
        raise ValueError(msg)

    max_contributions = max(contribution_counts)
    base_length_warn_threshold = 100
    # Parameters
    base_top_offset = 3.5
    base_width = 30
    base_height = 10
    bar_base_dimension = 2.5
    bar_max_height = 20
    bar_l_margin = 2.5
    bar_w_margin = 2.5

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
        translate([base_length - base_length / 5, base_height / 2 - base_top_offset / 2 - 1, -1.5])(
            linear_extrude(height=2)(text(str(year), 6))
        )
    )

    user_scad = rotate([base_face_angle, 0, 0])(
        translate([base_length / 4, base_height / 2 - base_top_offset / 2, -1.5])(
            linear_extrude(height=2)(text("@" + username, 5))
        )
    )

    script_path = Path(__file__).parent.absolute()
    logo_path = script_path / "gitlab.svg"

    logo_scad = rotate([base_face_angle, 0, 0])(
        translate([base_length / 8, base_height / 2 - base_top_offset / 2 - 2, -1])(
            linear_extrude(height=2)(scale([0.09, 0.09, 0.09])(import_stl(str(logo_path))))
        )
    )

    # Build bars
    bars = None

    week_number = 0
    last_weekday = 6  # Saturday
    for i in range(len(contribution_counts)):
        day_number = i % 7

        if contribution_counts[i] != 0:
            bar = translate(
                [
                    base_top_offset + bar_l_margin + week_number * bar_base_dimension,
                    base_top_offset + bar_w_margin + day_number * bar_base_dimension,
                    base_height,
                ]
            )(
                cube(
                    [
                        bar_base_dimension,
                        bar_base_dimension,
                        contribution_counts[i] / max_contributions * bar_max_height,
                    ]
                )
            )

            if bars is None:
                bars = bar
            else:
                bars += bar

        if day_number == last_weekday:
            week_number += 1

    scad_skyline_object = base_scad - logo_scad + user_scad + year_scad

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


def main():
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
    parser.add_argument("-o", "--output", type=Path, help="Output path", default=Path.cwd())
    parser.add_argument("--stl", help="Export an STL file as well (Requires openscad binary)", action="store_true")
    parser.add_argument("--domain", type=str, help="GitLab custom domain", default="https://gitlab.com")
    parser.add_argument("--token", type=str, help="Personal access token", default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Max concurrent requests to GitLab",
        default=2,
    )
    parser.add_argument("--loglevel", type=str, help="Log level", default="INFO")

    args = parser.parse_args()

    _init_logger()
    logger.setLevel(args.loglevel.upper())

    dates = get_dates_in_year(year=args.year)
    userid = get_userid(username=args.username, domain=args.domain)

    logger.info("Fetching contributions from GitLab...")
    date_contributions = get_contributions(
        userid=userid, dates=dates, domain=args.domain, token=args.token, concurrency=args.concurrency
    )

    ordered_contribution_counts = date_contributions_to_ordered_counts(date_contributions=date_contributions)
    logger.debug(f"Ordered contribution counts {ordered_contribution_counts}")
    contribution_counts = pad_contribution_counts_weekdays(
        ordered_contribution_counts=ordered_contribution_counts, first_date=dates[0], last_date=dates[-1]
    )
    logger.debug(f"Padded contribution counts: {contribution_counts}")

    logger.info("Generating model...")
    model = generate_skyline_stl(contribution_counts=contribution_counts, username=args.username, year=args.year)

    logger.info("Rendering models to file...")
    output_filename = f"gitlab_{args.username}_{args.year}"
    scad_path = args.output / f"{output_filename}.scad"
    stl_path = args.output / f"{output_filename}.stl"
    render_scad(model=model, path=scad_path)
    if args.stl:
        render_stl(scad_path=scad_path, path=stl_path)


if __name__ == "__main__":
    main()
