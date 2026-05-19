import json

def parse_text_column(row):
    try:
        data = json.loads(row['text'])
        row['type'] = data.get('type')

        prices = data.get('prices', [])
        for i, price_item in enumerate(prices):
            if i < 4:
                row[f'price{i+1}_value'] = price_item.get('price')
                row[f'price{i+1}_currency'] = price_item.get('currency')
                row[f'price{i+1}_sign'] = price_item.get('sign')
                row[f'price{i+1}_per'] = price_item.get('price_per')
                row[f'price{i+1}_is_discounted'] = price_item.get('price_is_discounted')
        
        origins = data.get('origins', [])
        row['origins'] = ', '.join(origins) if origins else None
        
        # Add other relevant fields if they exist
        row['organic'] = data.get('organic')
        row['barcode'] = data.get('barcode')
        row['product_name'] = data.get('product_name')
        row['category'] = data.get('category')
        row['has_multiple_categories'] = data.get('has_multiple_categories')
        row['uncertain_barcode_or_product_name'] = data.get('uncertain_barcode_or_product_name')
        row['is_price_tag'] = data.get('is_price_tag')

        return row
    except (json.JSONDecodeError, TypeError):
        return row
