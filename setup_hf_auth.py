#!/usr/bin/env python3
"""
Hugging Face Authentication Setup Guide
"""

print("=" * 60)
print("HUGGING FACE AUTHENTICATION SETUP")
print("=" * 60)

print("\nTo authenticate with Hugging Face and access Gemma models:")
print("\n1. Get your token:")
print("   - Go to: https://huggingface.co/settings/tokens")
print("   - Create a new token with 'Read' access")
print("   - Copy the token")

print("\n2. Set up authentication (choose ONE method):")

print("\n   METHOD A - Environment Variable (Recommended):")
print("   - Create a .env file in your project directory")
print("   - Add: HF_TOKEN=your_token_here")
print("   - The code will automatically use this")

print("\n   METHOD B - Login via CLI:")
print("   - Run: huggingface-cli login")
print("   - Paste your token when prompted")

print("\n   METHOD C - Login via Python:")
print("   - Run this script and follow prompts")

print("\n3. Verify access:")
print("   - The code will automatically check if you can access Gemma models")

print("\n" + "=" * 60)

# Option to login via Python
try:
    from huggingface_hub import login, whoami
    import getpass
    
    print("\nOPTION: Login via Python")
    choice = input("Do you want to login now via Python? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("\nPaste your Hugging Face token:")
        token = getpass.getpass("HF Token: ")
        
        try:
            login(token=token)
            user_info = whoami()
            print(f"\n✅ Successfully logged in as: {user_info['name']}")
            print("You can now use Gemma models!")
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            print("Please check your token and try again.")
    else:
        print("\nYou can set up authentication later using the methods above.")

except ImportError:
    print("\n❌ huggingface_hub not available")

print("\n" + "=" * 60)
