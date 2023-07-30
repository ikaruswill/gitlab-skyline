import argparse
import asyncio
import datetime
import logging
import math
import operator
import os
import subprocess
import sys
import urllib
from typing import List, Tuple

import aiohttp
import requests
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


async def get_contributions(
    semaphore: asyncio.Semaphore,
    domain: str,
    userid: int,
    token: str,
    date: datetime.date,
    date_contributions: List[Tuple[str, int]],
):
    """Get contributions directly using GitLab events API (asynchronously)"""

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
    pad_left_days = first_date.isoweekday() % 7  # Sun = 0, Mon = 1
    pad_right_days = 7 - last_date.isoweekday() % 7 + 1  # Sun = 6, Mon = 5
    left_padding = [0] * pad_left_days
    right_padding = [0] * pad_right_days
    return left_padding + ordered_contribution_counts + right_padding


def date_contributions_to_ordered_counts(date_contributions: List[Tuple[str, int]]) -> List[int]:
    """Sort by date and remove date from contributions"""
    sorted_date_contributions = sorted(date_contributions, key=operator.itemgetter(0))
    _, counts = list(zip(*sorted_date_contributions))
    return list(counts)


def generate_skyline_stl(contribution_counts: List[int], username: str, year: int):
    """Generate SCAD model of contributions"""
    max_contributions = max(contribution_counts)

    base_top_width = 23
    base_width = 30
    base_length = 150
    base_height = 10
    max_length_contributionbar = 20
    bar_base_dimension = 2.5

    # Build base
    base_top_offset = (base_width - base_top_width) / 2
    face_angle = math.degrees(math.atan(base_height / base_top_offset))

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

    year_scad = rotate([face_angle, 0, 0])(
        translate([base_length - base_length / 5, base_height / 2 - base_top_offset / 2 - 1, -1.5])(
            linear_extrude(height=2)(text(str(year), 6))
        )
    )

    user_scad = rotate([face_angle, 0, 0])(
        translate([base_length / 4, base_height / 2 - base_top_offset / 2, -1.5])(
            linear_extrude(height=2)(text("@" + username, 5))
        )
    )

    logo_gitlab_scad = rotate([face_angle, 0, 0])(
        translate([base_length / 8, base_height / 2 - base_top_offset / 2 - 2, -1])(
            linear_extrude(height=2)(
                scale([0.09, 0.09, 0.09])(
                    import_stl(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "gitlab.svg")
                )
            )
        )
    )

    # Build bars
    bars = None

    week_number = 1
    for i in range(len(contribution_counts)):
        day_number = i % 7
        if day_number == 0:
            week_number += 1

        if contribution_counts[i] == 0:
            continue

        bar = translate(
            [
                base_top_offset + 2.5 + (week_number - 1) * bar_base_dimension,
                base_top_offset + 2.5 + day_number * bar_base_dimension,
                base_height,
            ]
        )(
            cube(
                [
                    bar_base_dimension,
                    bar_base_dimension,
                    contribution_counts[i] * max_length_contributionbar / max_contributions,
                ]
            )
        )

        if bars is None:
            bars = bar
        else:
            bars += bar

    scad_skyline_object = base_scad - logo_gitlab_scad + user_scad + year_scad

    if bars is not None:
        scad_skyline_object += bars

    output_filename = f"gitlab_{username}_{year}"
    scad_render_to_file(scad_skyline_object, f"{output_filename}.scad")

    subprocess.run(
        ["openscad", "-o", f"{output_filename}.stl", f"{output_filename}.scad"],
        capture_output=True,
    )

    logger.info(f"Generated STL file: {output_filename}.stl ")


def main():
    _init_logger()
    logger.setLevel(args.loglevel.upper())

    all_dates = get_dates_in_year(year=args.year)
    userid = get_userid(username=args.username, domain=args.domain)

    date_contributions = []
    logger.info("Fetching contributions from GitLab...")
    semaphore = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[
                get_contributions(
                    semaphore=semaphore,
                    domain=args.domain,
                    userid=userid,
                    token=args.token,
                    date=date,
                    date_contributions=date_contributions,
                )
                for date in all_dates
            ]
        )
    )
    loop.close()

    ordered_contribution_counts = date_contributions_to_ordered_counts(date_contributions=date_contributions)
    contribution_counts = pad_contribution_counts_weekdays(
        ordered_contribution_counts=ordered_contribution_counts, first_date=all_dates[0], last_date=all_dates[1]
    )

    logger.info("Generating STL...")
    generate_skyline_stl(contribution_counts=contribution_counts, username=args.username, year=args.year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gitlab-skyline",
        description="Create STL from GitLab contributions",
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
    parser.add_argument("--domain", type=str, nargs="?", help="GitLab custom domain", default="https://gitlab.com")
    parser.add_argument("--token", type=str, nargs="?", help="Personal access token", default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Max concurrent requests to GitLab",
        default=2,
        nargs="?",
    )
    parser.add_argument("--loglevel", type=str, help="Log level", default="INFO")

    args = parser.parse_args()

    main()
