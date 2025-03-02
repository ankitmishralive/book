


from bs4 import BeautifulSoup
import requests
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}


def extract_book_details(url):
    """Extractingg book details from a single book page."""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.find("div", class_="product_main").find("h1").text.strip()
        price = soup.find("p", class_="price_color").text.strip()
        availability_text = soup.find("p", class_="instock availability").text.strip()

        description_element = soup.find("article", class_="product_page").find_all('p')
        if len(description_element) > 3:
            full_description = description_element[3].text.strip()
        else:
            full_description = "No description available"

        product_info = {}
        table = soup.find("table", class_="table table-striped")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) == 2:
                    key = row.find("th").text.strip()
                    value = cells[0].text.strip()
                    product_info[key] = value

        category = soup.find("ul", class_="breadcrumb").find_all("a")[2].text.strip()

        # Extractingg  specific product information
        upc = product_info.get("UPC")
        product_type = product_info.get("Product Type")
        price_excl_tax = product_info.get("Price (excl. tax)")
        price_incl_tax = product_info.get("Price (incl. tax)")
        tax = product_info.get("Tax")
        availability_number = product_info.get("Availability")
        number_of_reviews = product_info.get("Number of reviews")

        return {
            "title": title,
            "price": price,
            "availability": availability_text,
            "full_description": full_description,
            "category": category,
            "url": url,
            "UPC": upc,
            "Product Type": product_type,
            "Price (excl. tax)": price_excl_tax,
            "Price (incl. tax)": price_incl_tax,
            "Tax": tax,
            "Availability number": availability_number,
            "Number of reviews": number_of_reviews
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except AttributeError as e:
        print(f"AttributeError while parsing {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing {url}: {e}")
        return None

def scrape_all_books(base_url, max_pages=4):
    """Scraping  all books from the website, up to max_pages."""
    all_books = []
    page_number = 1
    while page_number <= max_pages:
        url = f"{base_url}catalogue/page-{page_number}.html" if page_number > 1 else f"{base_url}index.html"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            book_articles = soup.find_all("article", class_="product_pod")
            if not book_articles:
                break

            for article in book_articles:
                book_relative_url = article.find("a")["href"].replace("../", "")
                book_url = f"{base_url}catalogue/{book_relative_url.split('/')[-2]}/index.html"
                book_details = extract_book_details(book_url)
                if book_details:
                    all_books.append(book_details)
                time.sleep(1)

            page_number += 1
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    return all_books
