from Deck import Deck
from Card import Card, Suit, Rank
from Player import Player
from Trick import Trick

'''
Change auto to False if you would like to play the game manually.
This allows you to make all passes, and plays for all four players.
When auto is True, passing is disabled and the computer plays the
game by "guess and check", randomly trying moves until it finds a
valid one.
'''
auto = False

totalTricks = 13
maxScore = 100
queen = 12
noSuit = -1
clubs = 0
diamonds = 1
spades = 2
hearts = 3
cardsToPass = 3
numSuits = 4
suits = ["Clubs", "Diamonds", "Spades", "Hearts"]
num_cards = 52

class Hearts:
	def __init__(self):

		self.roundNum = 0
		self.trickNum = 0 # initialization value such that first round is round 0
		self.dealer = -1 # so that first dealer is 0
		self.passes = [1, -1, 2, 0] # left, right, across, no pass
		self.currentTrick = Trick()
		self.trickWinner = -1
		self.heartsBroken = False
		self.losingPlayer = None
		self.passingCards = [[], [], [], []]
		self.phase = "Pass"
		self.lastTrickPoints = 0


		# Make four players

		self.players = [Player("Danny"), Player("Desmond"), Player("Ben"), Player("Tyler")]

		'''
		Player physical locations:
		Game runs clockwise

			p3
		p2		p4
			p1

		'''

		# Generate a full deck of cards and shuffle it
		# Pretty sure reset() does this for us
		# self.newRound()

	def handleScoring(self):
		p, highestScore = None, 0
		for player in self.players:
			if player.roundScore == 26:
				p = player
				player.roundScore = 0
				break

		if p is not None:
			print(p.name + " shot the moon!")
			for player in self.players:
				if player != p:
					player.roundScore = 26
			p = None

		for player in self.players:
			player.score += player.roundScore
			player.roundScore = 0

		# print("\nScores:\n")
		for player in self.players:
			# print(player.name + ": " + str(player.score))
			if player.score > highestScore:
				p = player
				highestScore = player.score
			self.losingPlayer = p

	def newRound(self):
		self.deck = Deck()
		self.deck.shuffle()
		self.roundNum += 1
		self.trickNum = 0
		self.trickWinner = -1
		self.heartsBroken = False
		self.dealer = (self.dealer + 1) % len(self.players)
		self.dealCards()
		self.currentTrick = Trick()
		self.lastTrickPoints = 0
		self.passingCards = [[], [], [], []]
		self.phase = "Pass"
		for p in self.players:
			p.discardTricks()

	def getFirstTrickStarter(self):
		for player in range(len(self.players)):
			if self.players[player].hand.hand[0]:
				self.trickWinner = player

	def printHand(self, player):
		self.players[player].printHand()

	def findIndex(self, card):
		return ((card.suit.iden * 13) + card.rank.rank - 2)

	def dealCards(self):
		i = 0
		while(self.deck.size() > 0):
			card = self.findIndex(self.deck.deal())
			self.players[i % len(self.players)].addCard(card)
			i += 1


	def evaluateTrick(self):
		self.trickWinner = self.currentTrick.winner
		p = self.players[self.trickWinner]
		self.lastTrickPoints = self.currentTrick.points
		p.trickWon(self.currentTrick)
		# self.printCurrentTrick()
		# print(p.name + " won the trick.")
		# print 'Making new trick'
		self.currentTrick = Trick()
		# print(self.currentTrick.suit)


	def passCards(self, cardChoice, playerNum):
		passTo = self.passes[self.roundNum % 4] # how far to pass cards
		passTo = (playerNum + passTo) % len(self.players) # the index to which cards are passed
		# remove card from player hand and add to passed cards
		self.passingCards[passTo].append(cardChoice)
		self.players[playerNum].removeCard(cardChoice)


	def distributePassedCards(self):
		for i,passed in enumerate(self.passingCards):
			for card in passed:
				self.players[i].addCard(card)
		# self.passingCards = [[], [], [], []]


	def printPassingCards(self):
		out = "[ "
		for passed in self.passingCards:
			out += "["
			for card in passed:
				out += card.__str__() + " "
			out += "] "
		out += " ]"
		return out

	# Modifying
	def playersPassCards(self, playerNum, cardChoice):

		# We don't pass every 4th hand
		if self.roundNum % 4 == 3:
			self.phase = "Play"

		else:
			# self.printPlayers()
			
			# print # spacing
			# self.printPlayer(playerNum)
			self.passCards(cardChoice, playerNum % len(self.players))

			# Check if everyone has passed cards, if not, continue to next player. 
			if all(len(cardsToPass) == 3 for cardsToPass in self.passingCards):
				# print(self.printPassingCards())
				self.distributePassedCards()
				# self.printPlayers()
				self.phase = "Play"
		
		self.getFirstTrickStarter()

	# To define illegal plays we must check the player's hand and see what cards they have, and if they're legal in the current situation. We return a list of all legal plays.
	def isLegal(self, curPlayer):
		# Create 52-card action mask
		actionMask = [0] * 52

		# Any card not in hand is illegal, else the cards start legal
		for index in range(len(self.players[curPlayer].hand.hand)):
			if self.players[curPlayer].hand.hand[index]:
				actionMask[index] = 1

		# If a card is in the passingCards array, it is not a legal choice
		if self.phase == "Pass":
			for aCard in self.passingCards[curPlayer]:
				actionMask[aCard] = 0

		else:
			# If it's the first trick of a new hand, the card played must be the 2 of clubs
			if self.trickNum == 0 and self.currentTrick.cardsInTrick == 0:
				for index in range(1, len(actionMask)):
					actionMask[index] = 0
				
				return actionMask

			# If player only has hearts but hearts have not been broken, player can play hearts -- otherwise not
			if not self.heartsBroken:
				if self.players[curPlayer].hasOnlyHearts():
					for index in range(39, 52):
						if self.players[curPlayer].hand.hand[index]:
							actionMask[index] = 1
				else:
					for index in range(39, 52):
						actionMask[index] = 0

				
			# Card must be in currentTrick's suit if player has it
			if self.players[curPlayer].hasSuit(self.currentTrick.suit):

				if self.currentTrick.suit == 0:
					for index in range(13, len(actionMask)):
						actionMask[index] = 0

				elif self.currentTrick.suit == 1:
					for index in range(0, 13):
						actionMask[index] = 0
					for index in range(26, len(actionMask)):
						actionMask[index] = 0
						
				elif self.currentTrick.suit == 2:
					for index in range(0, 26):
						actionMask[index] = 0
					for index in range(39, len(actionMask)):
						actionMask[index] = 0

				elif self.currentTrick.suit == 3:
					for index in range(0, 39):
						actionMask[index] = 0

			# Hearts and the Queen of Spades cannot be broken on the first hand
			if self.trickNum == 0 and not self.players[curPlayer].startingHasOnlyHearts():
				for index in range(39, 52):
						actionMask[index] = 0

				# Queen of Spades
				actionMask[36] = 0

		return actionMask


	def playTrick(self, curPlayer, addCard):
		# If the played card is a heart or QoS and hearts aren't broken, break them
		if self.phase == "Play" and not self.heartsBroken and (addCard >= 39 and addCard < 52):
			self.heartsBroken = True

		# If it's the first play of a new trick
		if self.currentTrick.cardsInTrick == 0:
			self.players[curPlayer].removeCard(addCard)
			self.currentTrick.addCard(addCard, curPlayer)
			self.currentTrick.setTrickSuit(addCard)
			# self.printCurrentTrick()

		# Else if we're playing into an established trick
		else:
			self.players[curPlayer].removeCard(addCard)
			self.currentTrick.addCard(addCard, curPlayer)
			# self.printCurrentTrick()

		# If everyone has played, evaluate trick
		if self.currentTrick.cardsInTrick == 4:
			# self.printCurrentTrick()
			self.evaluateTrick()
			self.trickNum += 1

	# print all players' hands
	def printPlayers(self):
		for p in self.players:
			# print(p.name + ": " + str(p.hand))
			pass

	# show cards played in current trick
	def printCurrentTrick(self):
		trickStr = '\nCurrent table:\n'
		trickStr += "Trick suit: " + suits[self.currentTrick.suit] + "\n"
		for i, card in enumerate(self.currentTrick.trick):
			if self.currentTrick.trick[i] != -1:
				trickStr += self.players[i].name + ": " + str(card) + "\n"
			else:
				trickStr += self.players[i].name + ": None\n"
		print(trickStr)

	def getWinner(self):
		minScore = 200 # impossibly high
		winner = None
		for p in self.players:
			if p.score < minScore:
				winner = p
				minScore = p.score
		return winner
