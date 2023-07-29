#!/usr/bin/env python3

import argparse
import asyncio
import datetime
import logging
import math
import os
import subprocess
import sys
import urllib
from typing import List

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
    contribution_list: List[int],
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
                contribution_list[int(date.strftime("%j")) - 1] = len(json)

        except Exception as err:
            logger.error(f"Exception occured: {err}")
            pass


def get_dates_in_year(year: int) -> List[datetime.date]:
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year + 1, 1, 1)
    num_days = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=n) for n in range(num_days)]


def init_contribution_list(all_dates: List[datetime.date]) -> List[int]:
    first_date = all_dates[0]
    last_date = all_dates[-1]
    before_week_days = first_date.isoweekday() % 7  # Sun = 0, Mon = 1
    after_week_days = 7 - last_date.isoweekday() % 7 + 1  # Sun = 6, Mon = 5
    return [0] * before_week_days + [0] * len(all_dates) + [0] * after_week_days


def generate_skyline_stl(username, year, contribution_list):
    max_contributions = max(contribution_list)

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
    for i in range(len(contribution_list)):
        day_number = i % 7
        if day_number == 0:
            week_number += 1

        if contribution_list[i] == 0:
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
                    contribution_list[i] * max_length_contributionbar / max_contributions,
                ]
            )
        )

        if bars is None:
            bars = bar
        else:
            bars += bar

    scad_contributions_filename = "gitlab_" + username + "_" + str(year)
    scad_skyline_object = base_scad - logo_gitlab_scad + user_scad + year_scad

    if bars is not None:
        scad_skyline_object += bars

    scad_render_to_file(scad_skyline_object, scad_contributions_filename + ".scad")

    subprocess.run(
        ["openscad", "-o", scad_contributions_filename + ".stl", scad_contributions_filename + ".scad"],
        capture_output=True,
    )

    logger.info("Generated STL file " + scad_contributions_filename + ".stl")


def main():
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

    # Set logging
    _init_logger()
    logger.setLevel(args.loglevel.upper())

    all_dates = get_dates_in_year(args.year)
    contribution_list = init_contribution_list(all_dates)
    userid = get_userid(username=args.username, domain=args.domain)

    logger.info("Fetching contributions from GitLab...")

    semaphore = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[
                get_contributions(semaphore, args.domain, userid, args.token, date, contribution_list)
                for date in all_dates
            ]
        )
    )
    loop.close()

    logger.info("Generating STL...")
    generate_skyline_stl(args.username, args.year, contribution_list)


if __name__ == "__main__":
    main()
