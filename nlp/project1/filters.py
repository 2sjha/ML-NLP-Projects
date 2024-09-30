"""
Manual/Hand-picked filters to discard or allow items
Created by running individual scripts for the necessary task and observing the data
"""

import re
from utils import Page

NUM_CLEAN_FILES = 29

# List of initial links with further crawl = True
INITIAL_LINKS = [
    Page(
        "https://www.looper.com/161945/the-entire-story-of-the-office-finally-explained",
        True,
    ),
    Page(
        "https://www.britannica.com/topic/The-Office-American-television-program", True
    ),
    Page("https://theoffice.fandom.com/wiki/The_Office", True),
    Page(
        "http://warwickboar.shorthandstories.com/why--the-office-us--is-superior-to-the-uk-version",
        True,
    ),
    Page(
        "https://www.mentalfloss.com/article/60370/21-things-you-might-not-have-known-about-office",
        True,
    ),
]


BLOCK_LIST = set(
    [
        "https://www.threads.net",
        "https://www.static.com",
        "http://www.facebook.com",
        "https://www.facebook.com",
        "https://twitter.com",
        "https://www.youtube.com",
        "https://www.instagram.com",
        "https://flipboard.com",
        "https://story.snapchat.com",
        "https://news.google.com",
        "https://shorthand.com",
        "https://premium.britannica.com",
        "https://cdn.britannica.com",
        "https://www.merriam-webster.com",
        "https://bit.ly",
        "https://fandom.zendesk.com",
        "https://www.linkedin.com",
        "https://apps.apple.com",
        "http://en.wikipedia.com",
        "https://auth.fandom.com",
        "https://static.wikia.nocookie.net",
        "https://books.google.com",
        "https://www.tiktok.com",
        "https://www.amazon.com",
        "https://play.google.com",
        "https://about.fandom.com",
        "https://kids.britannica.com",
        "http://en.wikipedia.org",
        "https://www.fanatical.com",
        "https://ca.news.yahoo.com",
        "https://uk.news.yahoo.com",
        "https://www.muthead.com",
        "https://imgur.com",
        "https://themagickitchen.blogspot.com",
        "https://teamcoco.com",
        "http://www.alfredoscafe.com",
        "http://theofficestaremachine.com",
        "https://www.tripadvisor.com",
        "http://www.washingtonpost.com",
        "https://www.huffpost.com",
        "http://www.huffingtonpost.com",
        "https://www.simonandschuster.com",
        "http://articles.orlandosentinel.com",
        "https://news.yahoo.com",
        "https://www.minutemedia.com",
        "https://poorrichardspub.net",
        "https://www.coopers-seafood.com",
        "http://www.nytimes.com",
        "https://tv.avclub.com",
        "http://www.bnd.com",
        "http://screenertv.com",
        "http://heywriterboy.blogspot.com",
        "http://insidetv.ew.com",
    ]
)

SAME_DOMAIN_BLOCK_LIST = [
    re.compile(r"https://www.fandom.com/"),
    re.compile(r"https://www.fandom.com/terms-of-use"),
    re.compile(r"https://www.fandom.com/what-is-fandom"),
    re.compile(r"https://www.fandom.com/careers"),
    re.compile(r"https://www.fandom.com/video"),
    re.compile(r"https://www.fandom.com/explore"),
    re.compile(r"https://www.fandom.com/licensing"),
    re.compile(r"https://www.fandom.com/press"),
    re.compile(r"https://www.fandom.com/about#contact"),
    re.compile(r"https://www.fandom.com/privacy-policy"),
    re.compile(r"https://www.fandom.com/about"),
    re.compile(r"https://www.fandom.com/do-not-sell-my-info"),
    re.compile(r"https://www.fandom.com/topics/*"),
    re.compile(r"https://fandom.com/fancentral/home"),
    re.compile(r"https://theoffice.fandom.com/wiki/Special:AllMaps"),
    re.compile(r"https://theoffice.fandom.com/wiki/Mentioned*"),
    re.compile(r"https://theoffice.fandom.com/wiki/Category*"),
    re.compile(r"https://theoffice.fandom.com/wiki/Season_*"),
    re.compile(r"https://theoffice.fandom.com/wiki/Main_Page"),
    re.compile(r"https://theoffice.fandom.com/wiki/Special*"),
    re.compile(r"https://theoffice.fandom.com/wiki/Blog:*"),
    re.compile(r"https://theoffice.fandom.com/wiki/Dunderpedia:*"),
    re.compile(r"http://theoffice.wikia.com/wiki/Mentioned*"),
    re.compile(r"https://www.britannica.com/study/infographics"),
    re.compile(r"https://www.britannica.com/explore/*"),
    re.compile(r"https://www.britannica.com/study/*"),
    re.compile(r"https://www.britannica.com/technology/*"),
    re.compile(r"https://www.britannica.com/art/*"),
    re.compile(r"https://www.britannica.com/dictionary/*"),
    re.compile(r"https://www.britannica.com/plant/*"),
    re.compile(
        r"https://www.britannica.com/topic/The-Office-American-television-program/images-videos"
    ),
    re.compile(
        r"https://www.britannica.com/topic/The-Office-American-television-program/additional-info"
    ),
    re.compile(r"https://www.mentalfloss.com"),
    re.compile(r"https://www.mentalfloss.com/section/*"),
    re.compile(r"https://www.mentalfloss.com/biographies"),
    re.compile(r"http://www.hollywoodreporter.com/gallery/*"),
]

DISCARD_WORDS_LIST = [
    "https://",
    "http://",
    "Seasons: ",
    "Image: ",
    "Beth Lee",
    "See Also: ",
    "All Rights Reserved",
    "All rights reserved",
    "purchases made from links",
    "Our editors will review what",
    "Sign In",
    "in our Privacy Policy",
    "Subscribe to TV Guide",
    "Dune: ",
    "Oppenheimer",
    "Great Classic Films",
    "25 Best Comedies",
    "Marvel",
    "Karate Kid",
    "Most Anticipated Documentaries",
    "Deadpool",
    "Ben-Adir",
    "Stream on Hulu",
    "Hearst",
    "We may earn commission from links ",
    "The Office Recap",
    "Can Ed Helms Ride The Hangover",
    "Preppiness Explosion ",
    "you buy through our links",
    "straight from Elite Daily",
    "MORE: Discomfort Zone: ",
    "MORE: Cheat Sheet:",
    "MORE: The Housing Upturn",
    "TIMEBusiness",
    "From left: ",
    "DIGITAL SPY",
]


IMPORTANT_TERMS = [
    "Office",
    "Dunder Mifflin",
    "Scranton",
    "Pennsylvania",
    "Ryan Howard",
    "Michael Scott",
    "Meredith Palmer",
    "Christmas",
    "Carol Stills",
    "Jan Levinson",
    "Dwight Schrute",
    "Angela Martin",
    "Jim Halpert",
    "Pam Beesly",
    "Roy Anderson",
    "Stamford",
    "Oscar Martinez",
    "Andy Bernard",
    "Karen Filippelli",
    "Phyllis Lapin",
    "Bob Vance",
    "David Wallace",
    "Toby Flenderson",
]

PEOPLE = [
    "Ryan Howard",
    "Michael Scott",
    "Meredith Palmer",
    "Carol Stills",
    "Jan Levinson",
    "Dwight Schrute",
    "Angela Martin",
    "Jim Halpert",
    "Pam Beesly",
    "Roy Anderson",
    "Oscar Martinez",
    "Andy Bernard",
    "Karen Filippelli",
    "Phyllis Lapin",
    "Bob Vance",
    "David Wallace",
    "Toby Flenderson",
]

OTHERS = [
    "The Office (US)",
    "Dunder Mifflin",
    "Scranton",
    "Pennsylvania",
    "Christmas",
    "Stamford",
]
