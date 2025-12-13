pragma solidity >=0.5.11;
contract ERC20Interface {
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
}
contract IERC20Interface {
    function allowance(address owner, address spender) external view returns (uint256);
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function balanceOf(address account) external view returns (uint256);
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
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
  bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
  bool public paused;
  function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
  address payable public owner;
  function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
  uint256 public nextListingId;
  function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
  mapping (uint256 => Listing) public listingsById;
  function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
  mapping (uint256 => Purchase) public purchasesById;
  function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
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
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event Listed(uint256 id, uint256 pricePerToken, uint256 initialAmount, address seller);
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event Canceled(uint256 id);
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
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
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
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
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
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
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function cancelListing(uint256 id) external {
        Listing storage listing = listingsById[id];
        require(listing.active, "This listing was turned inactive already!");
        require(listing.seller == msg.sender || owner == msg.sender, "Only the listing owner or the contract owner can cancel the listing!");
        listing.active = false;
        emit Canceled(id);
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function setPaused(bool value) external onlyContractOwner {
        paused = value;
    }
function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
    function withdrawFunds(uint256 withdrawAmount) external onlyContractOwner {
        owner.transfer(withdrawAmount);
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
    function kill() external onlyContractOwner {
        selfdestruct(owner);
    }
function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
}