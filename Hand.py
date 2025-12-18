clubs = 0
diamonds = 1
spades = 2
hearts = 3
suits = ["c", "d", "s", "h"]
queen = 36

class Hand:
	def __init__(self):
		# create hand of cards split up by suit
		# Order is Clubs -> Diamonds -> Spades -> Hearts
		self.hand = [0] * 52
		self.contains2ofclubs = False

	# Determines current size of hand
	def size(self):
		counter = 0

		for aCard in self.hand:
			if aCard:
				counter += 1

		return counter

	def findIndex(self, card):
		return ((card.suit.iden * 13) + card.rank.rank - 2)			

	# Adds card to hand based on index value
	def addCard(self, addCard):

		# addCard = self.findIndex(card)

#		if addCard >= 0 and addCard < 13:
#			if addCard == 0:
#				self.contains2ofclubs = True
#			self.hand[addCard - 2] = 1
#		elif addCard >= 13 and addCard < 26:
#			self.hand[addCard + 11] = 1
#		elif addCard >= 26 and addCard < 39:
#			self.hand[addCard + 24] = 1
#		elif addCard >= 39 and addCard < 52:
#			self.hand[addCard + 37] = 1
#		else:
#			print('Invalid card')


		if addCard == 0:
			self.contains2ofclubs = True

		self.hand[addCard] = 1

		if addCard > 51:
			print('Invalid card')

	# Converts chosen card string into index:
	def strToCard(self, card):
		if len(card) == 0: return None

		suit = card[len(card)-1].lower() # get the suit from the string

		try:
			suitIden = suits.index(suit)
		except:
			print('Invalid suit')
			return None

		cardRank = card[0:len(card)-1] # get rank from string

		try:
			cardRank = cardRank.upper()
		except AttributeError:
			pass

		# convert rank to int
		if cardRank == "J":
			cardRank = 11
		elif cardRank == "Q":
			cardRank = 12
		elif cardRank == "K":
			cardRank = 13
		elif cardRank == "A":
			cardRank = 14
		else:
			try:
				cardRank = int(cardRank)
			except:
				print("Invalid card rank.")
				return None

		return (cardRank - 2) + (suitIden * 13)

	# Boolean check to see if card is present in hand
	def containsCard(self, cardIndex):
		if self.hand[cardIndex]:
			return self.hand[cardIndex]
		
		return False

	# Converts the card string to index and checks if the hand contains it
	def playCard(self, card):
		cardIndex = self.strToCard(card)

		if cardIndex is None:
			return None

		# see if player has that card in hand
		return self.containsCard(cardIndex)

	# Removes card from hand
	def removeCard(self, card):
		if card == 0:
			self.contains2ofclubs = False
			# print "Removing:", c.__str__()
		self.hand[card] = 0

	# Identifies if hand only contains hearts
	def hasOnlyHearts(self):
		hearts = self.hand[39:53]

		numHearts = hearts.count(1)

		# print("len(self.hearts):",numHearts)
		# print("self.size():",self.size())
		return numHearts == self.size()
	
	# Identifies if starting hand only contains hearts and queen of spades
	def startingHasOnlyHearts(self):
		hearts = self.hand[39:53]

		numHearts = hearts.count(1)

		if self.hand[36]:
			numHearts += 1

		# print("len(self.hearts):",numHearts)
		# print("self.size():",self.size())
		return numHearts == self.size()


	# Prints all cards in hand
	def __str__(self):
		handStr = ''
		for index in range(len(self.hand)):
			if self.hand[index] == 1:
				suitID = index // 13
				rank = (index % 13) + 2
				suitID = suits[suitID]
				if rank > 10:
					if rank == 11:
						rank = "J"
					elif rank == 12:
						rank = "Q"
					elif rank == 13:
						rank = "K"
					else:
						rank = "A"

				handStr += str(rank) + suitID + ' '
		return handStr
