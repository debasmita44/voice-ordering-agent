from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import os
from datetime import datetime
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Restaurant Configuration
RESTAURANT_NAME = "Twilight Cafe"
ASSISTANT_NAME = "Plato"

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Configure generation settings for better responses
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 200,
    }
    
    model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
else:
    print("WARNING: GEMINI_API_KEY not found. Please set it in environment variables.")
    model = None

# Menu items with prices
MENU = {
    'burger': {'name': 'Burger', 'price': 8.99},
    'cheeseburger': {'name': 'Cheeseburger', 'price': 9.99},
    'pizza': {'name': 'Pizza', 'price': 12.99},
    'pasta': {'name': 'Pasta', 'price': 10.99},
    'salad': {'name': 'Salad', 'price': 7.99},
    'fries': {'name': 'Fries', 'price': 3.99},
    'chicken wings': {'name': 'Chicken Wings', 'price': 11.99},
    'sandwich': {'name': 'Sandwich', 'price': 6.99},
    'soda': {'name': 'Soda', 'price': 2.99},
    'water': {'name': 'Water', 'price': 1.99},
    'coffee': {'name': 'Coffee', 'price': 3.49},
    'milkshake': {'name': 'Milkshake', 'price': 5.99}
}

# Store cart in memory
carts = {}
conversation_history = {}

def is_greeting_or_casual(text):
    """Check if text is just a greeting or casual conversation"""
    casual_patterns = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^good morning$', r'^good afternoon$',
        r'^good evening$', r'^how are you$', r'^thanks$', r'^thank you$',
        r'^okay$', r'^ok$', r'^yes$', r'^no$', r'^sure$', r'^alright$',
        r'^please$'
    ]
    
    text_lower = text.lower().strip()
    for pattern in casual_patterns:
        if re.match(pattern, text_lower):
            return True
    return False

def extract_order_with_gemini(user_text, conversation_context=""):
    """Use Gemini to extract order items and quantities"""
    
    if is_greeting_or_casual(user_text):
        return []
    
    if not model:
        return fallback_extract_order(user_text)
    
    menu_items_list = list(MENU.keys())
    menu_text = ", ".join(menu_items_list)
    
    prompt = f"""Extract food items and quantities from this order.

Available items: {menu_text}

Customer said: "{user_text}"

Rules:
- Match items even with "a", "an", "the" prefixes
- "a burger" = burger
- "two burgers" = 2 burgers
- Default quantity is 1
- Return empty array [] if no valid items found

Return ONLY valid JSON array, no markdown, no explanation:
[{{"item": "exact_menu_item", "quantity": number}}]

Examples:
"a burger" -> [{{"item": "burger", "quantity": 1}}]
"two burgers and pizza" -> [{{"item": "burger", "quantity": 2}}, {{"item": "pizza", "quantity": 1}}]
"hello" -> []

JSON response:"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Extract JSON
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        items = json.loads(response_text)
        
        # Validate and normalize items
        valid_items = []
        for item in items:
            item_name = item.get('item', '').lower().strip()
            quantity = item.get('quantity', 1)
            
            # Try exact match first
            if item_name in MENU:
                valid_items.append({
                    'key': item_name,
                    'name': MENU[item_name]['name'],
                    'price': MENU[item_name]['price'],
                    'quantity': quantity
                })
            else:
                # Try fuzzy match
                for menu_key in MENU.keys():
                    if menu_key in item_name or item_name in menu_key:
                        valid_items.append({
                            'key': menu_key,
                            'name': MENU[menu_key]['name'],
                            'price': MENU[menu_key]['price'],
                            'quantity': quantity
                        })
                        break
        
        return valid_items
    
    except Exception as e:
        print(f"Gemini extraction error: {e}")
        return fallback_extract_order(user_text)

def fallback_extract_order(user_text):
    """Simple fallback order extraction using regex"""
    items = []
    text_lower = user_text.lower()
    
    # Common quantity words
    quantity_map = {
        'a': 1, 'an': 1, 'one': 1,
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    for menu_key in MENU.keys():
        # Check if item is in text
        if menu_key in text_lower:
            quantity = 1
            
            # Look for quantity before the item
            pattern = r'(\w+)\s+' + menu_key
            match = re.search(pattern, text_lower)
            if match:
                qty_word = match.group(1)
                if qty_word in quantity_map:
                    quantity = quantity_map[qty_word]
                elif qty_word.isdigit():
                    quantity = int(qty_word)
            
            items.append({
                'key': menu_key,
                'name': MENU[menu_key]['name'],
                'price': MENU[menu_key]['price'],
                'quantity': quantity
            })
    
    return items

def generate_response_with_gemini(cart_items, added_items, total, action='add', user_text='', conversation_context=''):
    """Generate natural response using Gemini"""
    
    if not model:
        return get_fallback_response(action, added_items, total)
    
    cart_summary = ""
    if cart_items:
        cart_summary = ", ".join([f"{item['quantity']} {item['name']}" for item in cart_items])
    
    if action == 'welcome':
        prompt = f"""You're {ASSISTANT_NAME}, a friendly server at {RESTAURANT_NAME}. Write a warm 2-sentence welcome:
1. Greet them to {RESTAURANT_NAME}
2. Ask what they'd like

Be casual, use contractions. Example: "Hey! Welcome to {RESTAURANT_NAME}! I'm {ASSISTANT_NAME}. What can I get you today?"

Response (just the greeting, no quotes):"""
    
    elif action == 'greeting':
        prompt = f"""You're {ASSISTANT_NAME}, a server at {RESTAURANT_NAME}.

Customer: "{user_text}"

Respond in 1 sentence. Be friendly, then ask what they want to order.

Example: "Hey! What sounds good to you?"

Response:"""
    
    elif action == 'add' and added_items:
        items_text = ', '.join([f"{item['quantity']} {item['name']}" for item in added_items])
        
        prompt = f"""You're {ASSISTANT_NAME}, a server at {RESTAURANT_NAME}.

Just added: {items_text}
Cart now has: {cart_summary}
Total: ${total:.2f}

Write 2-3 sentences:
1. Confirm what was added (be enthusiastic!)
2. Say the total
3. Ask if they want more

Be casual, conversational. Use contractions.

Example: "Nice! Got your {items_text}. That's ${total:.2f} so far. Want anything else?"

Response (no quotes):"""
    
    elif action == 'no_items':
        prompt = f"""You're {ASSISTANT_NAME}, a server at {RESTAURANT_NAME}.

Customer: "{user_text}"

You didn't understand their order. Politely ask them to repeat (1 sentence).

Example: "Sorry, didn't catch that! What would you like?"

Response:"""
    
    elif action == 'checkout':
        prompt = f"""You're {ASSISTANT_NAME}, a server at {RESTAURANT_NAME}.

Customer is checking out.
Total: ${total:.2f}

Write 2-3 sentences:
1. Thank them
2. Confirm total
3. Say food will be ready soon

Be warm and appreciative.

Example: "Awesome! Thanks for ordering. Your total is ${total:.2f}. We'll have that ready in a few minutes!"

Response (no quotes):"""
    
    else:
        return "What can I get for you?"
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        # Remove quotes if present
        result = result.strip('"\'')
        return result
    
    except Exception as e:
        print(f"Gemini response error: {e}")
        return get_fallback_response(action, added_items, total)

def get_fallback_response(action, added_items=None, total=0):
    """Fallback responses when Gemini is unavailable"""
    if action == 'welcome':
        return f"Hey! Welcome to {RESTAURANT_NAME}! I'm {ASSISTANT_NAME}. What can I get you today?"
    elif action == 'greeting':
        return "Hey! What sounds good to you?"
    elif action == 'add' and added_items:
        items_text = ', '.join([f"{item['quantity']} {item['name']}" for item in added_items])
        return f"Nice! Got your {items_text}. That's ${total:.2f} so far. Want anything else?"
    elif action == 'no_items':
        return "Sorry, didn't catch that! What would you like?"
    elif action == 'checkout':
        return f"Awesome! Thanks for ordering. Your total is ${total:.2f}. We'll have that ready soon!"
    return "What can I get for you?"

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'restaurant': RESTAURANT_NAME,
        'assistant': ASSISTANT_NAME,
        'gemini_configured': bool(GEMINI_API_KEY)
    })

@app.route('/api/menu', methods=['GET'])
def get_menu():
    """Get full menu"""
    return jsonify({'menu': MENU})

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get restaurant configuration"""
    return jsonify({
        'restaurant_name': RESTAURANT_NAME,
        'assistant_name': ASSISTANT_NAME
    })

@app.route('/api/process-order', methods=['POST'])
def process_order():
    """Process voice order and update cart"""
    data = request.json
    user_text = data.get('text', '')
    session_id = data.get('session_id', 'default')
    
    print(f"Processing order - User said: '{user_text}'")
    
    if session_id not in carts:
        carts[session_id] = []
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append(f"Customer: {user_text}")
    
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]
    
    context = "\n".join(conversation_history[session_id][-6:])
    lower_text = user_text.lower()
    
    # Check if greeting
    if is_greeting_or_casual(user_text):
        response_text = generate_response_with_gemini([], [], 0, action='greeting', user_text=user_text, conversation_context=context)
        conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
        
        return jsonify({
            'success': True,
            'cart': carts[session_id],
            'total': sum(item['price'] * item['quantity'] for item in carts[session_id]),
            'response': response_text,
            'items_added': [],
            'is_greeting': True
        })
    
    # Clear cart
    if any(word in lower_text for word in ['clear', 'empty cart', 'remove everything', 'reset cart']):
        carts[session_id] = []
        response_text = "Cart cleared! What would you like to order?"
        conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
        
        return jsonify({
            'success': True,
            'cart': [],
            'total': 0,
            'response': response_text,
            'items_added': []
        })
    
    # Checkout
    checkout_phrases = [
        'checkout', 'check out', 'complete order', 'complete my order', 
        'finish order', 'done', "that's all", "that is all", 'finish', 
        'place order', 'place my order', 'complete', "i'm done", "im done"
    ]
    
    if any(phrase in lower_text for phrase in checkout_phrases):
        if not carts[session_id]:
            response_text = "Your cart's empty! What would you like?"
            conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
            
            return jsonify({
                'success': False,
                'cart': [],
                'total': 0,
                'response': response_text,
                'checkout': False
            })
        
        total = sum(item['price'] * item['quantity'] for item in carts[session_id])
        response_text = generate_response_with_gemini(carts[session_id], [], total, action='checkout', conversation_context=context)
        conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
        
        return jsonify({
            'success': True,
            'cart': carts[session_id],
            'total': total,
            'response': response_text,
            'checkout': True
        })
    
    # Extract items
    extracted_items = extract_order_with_gemini(user_text, context)
    
    print(f"Extracted items: {extracted_items}")
    
    if not extracted_items:
        response_text = generate_response_with_gemini([], [], 0, action='no_items', user_text=user_text, conversation_context=context)
        conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
        
        return jsonify({
            'success': False,
            'cart': carts[session_id],
            'total': sum(item['price'] * item['quantity'] for item in carts[session_id]),
            'response': response_text,
            'items_added': []
        })
    
    # Add items to cart
    for item in extracted_items:
        existing_item = next((x for x in carts[session_id] if x['key'] == item['key']), None)
        if existing_item:
            existing_item['quantity'] += item['quantity']
        else:
            carts[session_id].append(item)
    
    total = sum(item['price'] * item['quantity'] for item in carts[session_id])
    response_text = generate_response_with_gemini(carts[session_id], extracted_items, total, action='add', conversation_context=context)
    conversation_history[session_id].append(f"{ASSISTANT_NAME}: {response_text}")
    
    print(f"Cart updated: {carts[session_id]}")
    print(f"Response: {response_text}")
    
    return jsonify({
        'success': True,
        'cart': carts[session_id],
        'total': total,
        'response': response_text,
        'items_added': extracted_items
    })

@app.route('/api/cart/<session_id>', methods=['GET'])
def get_cart(session_id):
    """Get current cart"""
    cart = carts.get(session_id, [])
    total = sum(item['price'] * item['quantity'] for item in cart)
    return jsonify({
        'cart': cart,
        'total': total
    })

@app.route('/api/welcome', methods=['GET'])
def get_welcome():
    """Get welcome message"""
    response_text = generate_response_with_gemini([], [], 0, action='welcome')
    return jsonify({
        'response': response_text
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
