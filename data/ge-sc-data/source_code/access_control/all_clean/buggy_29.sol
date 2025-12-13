pragma solidity >=0.5.11;
contract ERC20Interface {
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
  require(tx.origin == owner_txorigin7);
  to.call.value(amount);
}
}
contract IERC20Interface {
    function allowance(address owner, address spender) external view returns (uint256);
function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
  require(tx.origin == owner_txorigin23);
  to.call.value(amount);
}
    function balanceOf(address account) external view returns (uint256);
function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
}
contract RaffleToken is ERC20Interface, IERC20Interface {}
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "SafeMath: modulo by zero");
        return a % b;
    }
}
contract RaffleTokenExchange {
    using SafeMath for uint256;
    RaffleToken constant public raffleContract = RaffleToken(0x0C8cDC16973E88FAb31DD0FCB844DdF0e1056dE2);
  function transferTo_txorigin19(address to, uint amount,address owner_txorigin19) public {
  require(tx.origin == owner_txorigin19);
  to.call.value(amount);
}
  bool public paused;
  function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
        require(tx.origin == owner_txorigin26);
        _recipient.transfer(address(this).balance);
    }
  address payable public owner;
  function bug_txorigin20(address owner_txorigin20) public{
        require(tx.origin == owner_txorigin20);
    }
  uint256 public nextListingId;
  function bug_txorigin32(  address owner_txorigin32) public{
        require(tx.origin == owner_txorigin32);
    }
  mapping (uint256 => Listing) public listingsById;
  function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
        require(tx.origin == owner_txorigin38);
        _recipient.transfer(address(this).balance);
    }
  mapping (uint256 => Purchase) public purchasesById;
  function bug_txorigin4(address owner_txorigin4) public{
        require(tx.origin == owner_txorigin4);
    }
  uint256 public nextPurchaseId;
    struct Listing {
        uint256 pricePerToken;
        uint256 initialAmount;
        uint256 amountLeft;
        address payable seller;
        bool active;
    }
    struct Purchase {
        uint256 totalAmount;
        uint256 totalAmountPayed;
        uint256 timestamp;
    }
  function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
  require(tx.origin == owner_txorigin27);
  to.call.value(amount);
}
  event Listed(uint256 id, uint256 pricePerToken, uint256 initialAmount, address seller);
  function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
  require(tx.origin == owner_txorigin31);
  to.call.value(amount);
}
  event Canceled(uint256 id);
  function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
	require (tx.origin == owner_txorigin13);
	receiver.transfer(amount);
}
  event Purchased(uint256 id, uint256 totalAmount, uint256 totalAmountPayed, uint256 timestamp);
    modifier onlyContractOwner {
        require(msg.sender == owner, "Function called by non-owner.");
        _;
    }
    modifier onlyUnpaused {
        require(paused == false, "Exchange is paused.");
        _;
    }
    constructor() public {
        owner = msg.sender;
        nextListingId = 916;
        nextPurchaseId = 344;
    }
function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
        require(tx.origin == owner_txorigin30);
        _recipient.transfer(address(this).balance);
    }
    function buyRaffle(uint256[] calldata amounts, uint256[] calldata listingIds) payable external onlyUnpaused {
        require(amounts.length == listingIds.length, "You have to provide amounts for every single listing!");
        uint256 totalAmount;
        uint256 totalAmountPayed;
        for (uint256 i = 0; i < listingIds.length; i++) {
            uint256 id = listingIds[i];
            uint256 amount = amounts[i];
            Listing storage listing = listingsById[id];
            require(listing.active, "Listing is not active anymore!");
            listing.amountLeft = listing.amountLeft.sub(amount);
            require(listing.amountLeft >= 0, "Amount left needs to be higher than 0.");
            if(listing.amountLeft == 0) { listing.active = false; }
            uint256 amountToPay = listing.pricePerToken * amount;
            listing.seller.transfer(amountToPay);
            totalAmountPayed = totalAmountPayed.add(amountToPay);
            totalAmount = totalAmount.add(amount);
            require(raffleContract.transferFrom(listing.seller, msg.sender, amount), 'Token transfer failed!');
        }
        require(totalAmountPayed <= msg.value, 'Overpayed!');
        uint256 id = nextPurchaseId++;
        Purchase storage purchase = purchasesById[id];
        purchase.totalAmount = totalAmount;
        purchase.totalAmountPayed = totalAmountPayed;
        purchase.timestamp = now;
        emit Purchased(id, totalAmount, totalAmountPayed, now);
    }
function bug_txorigin8(address owner_txorigin8) public{
        require(tx.origin == owner_txorigin8);
    }
    function addListing(uint256 initialAmount, uint256 pricePerToken) external onlyUnpaused {
        require(raffleContract.balanceOf(msg.sender) >= initialAmount, "Amount to sell is higher than balance!");
        require(raffleContract.allowance(msg.sender, address(this)) >= initialAmount, "Allowance is to small (increase allowance)!");
        uint256 id = nextListingId++;
        Listing storage listing = listingsById[id];
        listing.initialAmount = initialAmount;
        listing.amountLeft = initialAmount;
        listing.pricePerToken = pricePerToken;
        listing.seller = msg.sender;
        listing.active = true;
        emit Listed(id, listing.pricePerToken, listing.initialAmount, listing.seller);
    }
function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
  require(tx.origin == owner_txorigin39);
  to.call.value(amount);
}
    function cancelListing(uint256 id) external {
        Listing storage listing = listingsById[id];
        require(listing.active, "This listing was turned inactive already!");
        require(listing.seller == msg.sender || owner == msg.sender, "Only the listing owner or the contract owner can cancel the listing!");
        listing.active = false;
        emit Canceled(id);
    }
function bug_txorigin36(  address owner_txorigin36) public{
        require(tx.origin == owner_txorigin36);
    }
    function setPaused(bool value) external onlyContractOwner {
        paused = value;
    }
function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
  require(tx.origin == owner_txorigin35);
  to.call.value(amount);
}
    function withdrawFunds(uint256 withdrawAmount) external onlyContractOwner {
        owner.transfer(withdrawAmount);
    }
function bug_txorigin40(address owner_txorigin40) public{
        require(tx.origin == owner_txorigin40);
    }
    function kill() external onlyContractOwner {
        selfdestruct(owner);
    }
function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
	require (tx.origin == owner_txorigin33);
	receiver.transfer(amount);
}
}