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
    model = genai.GenerativeModel('gemini-2.5-flash')
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

# Store cart in memory (in production, use a database)
carts = {}
conversation_history = {}

def is_greeting_or_casual(text):
    """Check if text is just a greeting or casual conversation"""
    casual_patterns = [
        r'^hi+$', r'^hello+$', r'^hey+$', r'^good morning$', r'^good afternoon$',
        r'^good evening$', r'^how are you$', r'^thanks$', r'^thank you$',
        r'^okay$', r'^ok$', r'^yes$', r'^no$', r'^sure$', r'^alright$',
        r'^please$', r'^please.*'
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
        return []
    
    menu_items = ', '.join(MENU.keys())
    
    context_info = ""
    if conversation_context:
        context_info = f"\n\nConversation context:\n{conversation_context}"
    
    prompt = f"""You are an AI order assistant. Extract ONLY actual food order items and quantities from the user's message.

Available menu items: {menu_items}

User said: "{user_text}"{context_info}

IMPORTANT RULES:
1. ONLY extract items that are clear food orders
2. IGNORE greetings like "hello", "hi", "hey", "please"
3. IGNORE casual responses like "yes", "no", "okay"
4. If the user is just chatting or greeting, return: []
5. Look for implicit orders like "I'll have", "give me", "can I get"

Return ONLY a JSON array with this exact format (no other text, no markdown):
[{{"item": "item_name", "quantity": number}}]

Examples:
- "I want two burgers and a pizza" -> [{{"item": "burger", "quantity": 2}}, {{"item": "pizza", "quantity": 1}}]
- "Can I get three fries" -> [{{"item": "fries", "quantity": 3}}]
- "please" -> []
- "hello hello" -> []

Response:"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        items = json.loads(response_text)
        
        # Validate items against menu
        valid_items = []
        for item in items:
            item_name = item.get('item', '').lower()
            quantity = item.get('quantity', 1)
            
            if item_name in MENU:
                valid_items.append({
                    'key': item_name,
                    'name': MENU[item_name]['name'],
                    'price': MENU[item_name]['price'],
                    'quantity': quantity
                })
        
        return valid_items
    
    except Exception as e:
        print(f"Gemini error: {e}")
        return []

def generate_response_with_gemini(cart_items, added_items, total, action='add', user_text='', conversation_context=''):
    """Generate natural response using Gemini"""
    
    if not model:
        return get_fallback_response(action, added_items, total)
    
    cart_summary = ""
    if cart_items:
        cart_summary = "Current cart: " + ", ".join([f"{item['quantity']} {item['name']}" for item in cart_items])
    
    context_info = ""
    if conversation_context:
        context_info = f"\n\nRecent conversation:\n{conversation_context}"
    
    if action == 'welcome':
        prompt = f"""You are {ASSISTANT_NAME}, a warm and friendly human server at {RESTAURANT_NAME}. 

Generate a natural welcome greeting (2-3 sentences) as if greeting a guest who just walked in. You should:
1. Welcome them warmly to {RESTAURANT_NAME}
2. Introduce yourself naturally as {ASSISTANT_NAME}
3. Ask what they would like to order conversationally

Speak like a real person, not a robot. Use contractions.

Example: "Hey there! Welcome to {RESTAURANT_NAME}! I'm {ASSISTANT_NAME}, and I'll be taking care of you today. What can I get started for you?"

Your response (no extra formatting):"""
    
    elif action == 'greeting':
        prompt = f"""You are {ASSISTANT_NAME}, a warm friendly server at {RESTAURANT_NAME}.

Customer said: "{user_text}"

This is casual conversation, not an order. Respond naturally (1-2 sentences) as a real server would:
- Greet them back warmly if they greet you
- If they say please/thanks, acknowledge kindly
- Then redirect to helping them order

{context_info}

Speak naturally. Use contractions.

Examples:
- "Hey! Good to see you! So what sounds good to you today?"
- "Of course! Happy to help. What are you in the mood for?"

Your response (no extra formatting):"""
    
    elif action == 'add' and added_items:
        items_text = ', '.join([f"{item['quantity']} {item['name']}" for item in added_items])
        
        prompt = f"""You are {ASSISTANT_NAME}, a warm friendly server at {RESTAURANT_NAME}.

Customer just ordered: {items_text}
{cart_summary}
New total: ${total:.2f}

Generate natural confirmation (2-3 sentences) like a real server:
1. Acknowledge their order enthusiastically 
2. Confirm what you added
3. Tell them the new total
4. Ask if they want anything else

{context_info}

Speak naturally with personality. Use contractions.

Examples:
- "Awesome choice! I've got your {items_text} coming right up. That brings you to ${total:.2f}. Anything else I can grab for you?"
- "Perfect! Added {items_text}. You're at ${total:.2f} so far. Want to add anything else?"

Your response (no extra formatting):"""
    
    elif action == 'no_items':
        prompt = f"""You are {ASSISTANT_NAME}, a warm friendly server at {RESTAURANT_NAME}.

Customer said: "{user_text}"

You didn't catch any valid menu items. Respond naturally (1-2 sentences):
- Politely say you didn't catch that
- Offer to help with the menu
- Keep it friendly

{context_info}

Examples:
- "Sorry, I didn't quite catch that! Could you tell me again what you'd like?"
- "Hmm, I'm not sure I got that right. What would you like to order?"

Your response (no extra formatting):"""
    
    elif action == 'checkout':
        prompt = f"""You are {ASSISTANT_NAME}, a warm friendly server at {RESTAURANT_NAME}.

Customer is ready to complete their order.
Final total: ${total:.2f}

Generate warm closing (2-3 sentences):
1. Thank them sincerely
2. Confirm the total
3. Let them know food will be ready soon

{context_info}

Examples:
- "Awesome! Thanks so much for your order. Your total comes to ${total:.2f}. We'll have that ready for you in just a few minutes!"
- "Perfect! That'll be ${total:.2f}. Thanks for ordering with us - your food will be right out!"

Your response (no extra formatting):"""
    
    else:
        return "How can I help you today?"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"Gemini response error: {e}")
        return get_fallback_response(action, added_items, total)

def get_fallback_response(action, added_items=None, total=0):
    """Fallback responses when Gemini is unavailable"""
    if action == 'welcome':
        return f"Hey there! Welcome to {RESTAURANT_NAME}! I'm {ASSISTANT_NAME}, and I'll be helping you today. What can I get started for you?"
    elif action == 'greeting':
        return "Hey! Good to see you! What sounds good to you today?"
    elif action == 'add' and added_items:
        items_text = ', '.join([f"{item['quantity']} {item['name']}" for item in added_items])
        return f"Awesome! I've got {items_text} for you. That's ${total:.2f} total. Anything else?"
    elif action == 'no_items':
        return "Sorry, I didn't catch that! What would you like to order?"
    elif action == 'checkout':
        return f"Perfect! Your total is ${total:.2f}. We'll have that ready for you soon. Thanks!"
    return "How can I help you?"

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
            response_text = "Your cart is empty! What would you like to order?"
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
