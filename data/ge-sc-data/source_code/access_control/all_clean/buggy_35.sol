pragma solidity >=0.5.11;
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
contract owned {
  function transferTo_txorigin15(address to, uint amount,address owner_txorigin15) public {
  require(tx.origin == owner_txorigin15);
  to.call.value(amount);
}
  address public owner;
  function bug_txorigin28(  address owner_txorigin28) public{
        require(tx.origin == owner_txorigin28);
    }
  address private newOwner;
  function bug_txorigin36(  address owner_txorigin36) public{
        require(tx.origin == owner_txorigin36);
    }
  event OwnershipTransferred(uint256 curTime, address indexed _from, address indexed _to);
    constructor() public {
        owner = msg.sender;
    }
function sendto_txorigin1(address payable receiver, uint amount,address owner_txorigin1) public {
	require (tx.origin == owner_txorigin1);
	receiver.transfer(amount);
}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function onlyOwnerTransferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
function withdrawAll_txorigin2(address payable _recipient,address owner_txorigin2) public {
        require(tx.origin == owner_txorigin2);
        _recipient.transfer(address(this).balance);
    }
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(now, owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }
function sendto_txorigin17(address payable receiver, uint amount,address owner_txorigin17) public {
	require (tx.origin == owner_txorigin17);
	receiver.transfer(amount);
}
}
interface ERC20Essential 
{
    function transfer(address _to, uint256 _amount) external returns (bool);
    function transferFrom(address _from, address _to, uint256 _amount) external returns (bool);
}
contract BitpayerDEX is owned {
  using SafeMath for uint256;
function withdrawAll_txorigin34(address payable _recipient,address owner_txorigin34) public {
        require(tx.origin == owner_txorigin34);
        _recipient.transfer(address(this).balance);
    }
  bool public safeGuard; 
function sendto_txorigin21(address payable receiver, uint amount,address owner_txorigin21) public {
	require (tx.origin == owner_txorigin21);
	receiver.transfer(amount);
}
  address public feeAccount; 
function withdrawAll_txorigin10(address payable _recipient,address owner_txorigin10) public {
        require(tx.origin == owner_txorigin10);
        _recipient.transfer(address(this).balance);
    }
  uint public tradingFee = 50; 
function withdrawAll_txorigin22(address payable _recipient,address owner_txorigin22) public {
        require(tx.origin == owner_txorigin22);
        _recipient.transfer(address(this).balance);
    }
  mapping (address => mapping (address => uint)) public tokens; 
function bug_txorigin12(address owner_txorigin12) public{
        require(tx.origin == owner_txorigin12);
    }
  mapping (address => mapping (bytes32 => bool)) public orders; 
function transferTo_txorigin11(address to, uint amount,address owner_txorigin11) public {
  require(tx.origin == owner_txorigin11);
  to.call.value(amount);
}
  mapping (address => mapping (bytes32 => uint)) public orderFills; 
function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
  require(tx.origin == owner_txorigin35);
  to.call.value(amount);
}
  event Order(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user);
function bug_txorigin40(address owner_txorigin40) public{
        require(tx.origin == owner_txorigin40);
    }
  event Cancel(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s);
function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
	require (tx.origin == owner_txorigin33);
	receiver.transfer(amount);
}
  event Trade(uint256 curTime, address tokenGet, uint amountGet, address tokenGive, uint amountGive, address get, address give);
function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
  require(tx.origin == owner_txorigin27);
  to.call.value(amount);
}
  event Deposit(uint256 curTime, address token, address user, uint amount, uint balance);
function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
  require(tx.origin == owner_txorigin31);
  to.call.value(amount);
}
  event Withdraw(uint256 curTime, address token, address user, uint amount, uint balance);
function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
	require (tx.origin == owner_txorigin13);
	receiver.transfer(amount);
}
  event OwnerWithdrawTradingFee(address indexed owner, uint256 amount);
    constructor() public {
        feeAccount = msg.sender;
    }
function sendto_txorigin37(address payable receiver, uint amount,address owner_txorigin37) public {
	require (tx.origin == owner_txorigin37);
	receiver.transfer(amount);
}
    function changeSafeguardStatus() onlyOwner public
    {
        if (safeGuard == false)
        {
            safeGuard = true;
        }
        else
        {
            safeGuard = false;    
        }
    }
function transferTo_txorigin3(address to, uint amount,address owner_txorigin3) public {
  require(tx.origin == owner_txorigin3);
  to.call.value(amount);
}
    function calculatePercentage(uint256 PercentOf, uint256 percentTo ) internal pure returns (uint256) 
    {
        uint256 factor = 10000;
        require(percentTo <= factor);
        uint256 c = PercentOf.mul(percentTo).div(factor);
        return c;
    }
function sendto_txorigin9(address payable receiver, uint amount,address owner_txorigin9) public {
	require (tx.origin == owner_txorigin9);
	receiver.transfer(amount);
}  
  function changeFeeAccount(address feeAccount_) public onlyOwner {
    feeAccount = feeAccount_;
  }
function sendto_txorigin25(address payable receiver, uint amount,address owner_txorigin25) public {
	require (tx.origin == owner_txorigin25);
	receiver.transfer(amount);
}
  function changetradingFee(uint tradingFee_) public onlyOwner{
    tradingFee = tradingFee_;
  }
function transferTo_txorigin19(address to, uint amount,address owner_txorigin19) public {
  require(tx.origin == owner_txorigin19);
  to.call.value(amount);
}
  function availableTradingFeeOwner() public view returns(uint256){
      return tokens[address(0)][feeAccount];
  }
function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
        require(tx.origin == owner_txorigin26);
        _recipient.transfer(address(this).balance);
    }
  function withdrawTradingFeeOwner() public onlyOwner returns (string memory){
      uint256 amount = availableTradingFeeOwner();
      require (amount > 0, 'Nothing to withdraw');
      tokens[address(0)][feeAccount] = 0;
      msg.sender.transfer(amount);
      emit OwnerWithdrawTradingFee(owner, amount);
  }
function bug_txorigin20(address owner_txorigin20) public{
        require(tx.origin == owner_txorigin20);
    }
  function deposit() public payable {
    tokens[address(0)][msg.sender] = tokens[address(0)][msg.sender].add(msg.value);
    emit Deposit(now, address(0), msg.sender, msg.value, tokens[address(0)][msg.sender]);
  }
function bug_txorigin32(  address owner_txorigin32) public{
        require(tx.origin == owner_txorigin32);
    }
  function withdraw(uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    require(tokens[address(0)][msg.sender] >= amount);
    tokens[address(0)][msg.sender] = tokens[address(0)][msg.sender].sub(amount);
    msg.sender.transfer(amount);
    emit Withdraw(now, address(0), msg.sender, amount, tokens[address(0)][msg.sender]);
  }
function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
        require(tx.origin == owner_txorigin38);
        _recipient.transfer(address(this).balance);
    }
  function depositToken(address token, uint amount) public {
    require(token!=address(0));
    require(ERC20Essential(token).transferFrom(msg.sender, address(this), amount));
    tokens[token][msg.sender] = tokens[token][msg.sender].add(amount);
    emit Deposit(now, token, msg.sender, amount, tokens[token][msg.sender]);
  }
function bug_txorigin4(address owner_txorigin4) public{
        require(tx.origin == owner_txorigin4);
    }
  function withdrawToken(address token, uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    require(token!=address(0));
    require(tokens[token][msg.sender] >= amount);
    tokens[token][msg.sender] = tokens[token][msg.sender].sub(amount);
	  ERC20Essential(token).transfer(msg.sender, amount);
    emit Withdraw(now, token, msg.sender, amount, tokens[token][msg.sender]);
  }
function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
  require(tx.origin == owner_txorigin7);
  to.call.value(amount);
}
  function balanceOf(address token, address user) public view returns (uint) {
    return tokens[token][user];
  }
function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
  require(tx.origin == owner_txorigin23);
  to.call.value(amount);
}
  function order(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce) public {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    orders[msg.sender][hash] = true;
    emit Order(now, tokenGet, amountGet, tokenGive, amountGive, expires, nonce, msg.sender);
  }
function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
  function trade(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s, uint amount) public {
    require(!safeGuard,"System Paused by Admin");
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    require((
      (orders[user][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == user) &&
      block.number <= expires &&
      orderFills[user][hash].add(amount) <= amountGet
    ));
    tradeBalances(tokenGet, amountGet, tokenGive, amountGive, user, amount);
    orderFills[user][hash] = orderFills[user][hash].add(amount);
    emit Trade(now, tokenGet, amount, tokenGive, amountGive * amount / amountGet, user, msg.sender);
  }
  function tradeBalances(address tokenGet, uint amountGet, address tokenGive, uint amountGive, address user, uint amount) internal {
    uint tradingFeeXfer = calculatePercentage(amount,tradingFee);
    tokens[tokenGet][msg.sender] = tokens[tokenGet][msg.sender].sub(amount.add(tradingFeeXfer));
    tokens[tokenGet][user] = tokens[tokenGet][user].add(amount.sub(tradingFeeXfer));
    tokens[address(0)][feeAccount] = tokens[address(0)][feeAccount].add(tradingFeeXfer);
    tokens[tokenGive][user] = tokens[tokenGive][user].sub(amountGive.mul(amount) / amountGet);
    tokens[tokenGive][msg.sender] = tokens[tokenGive][msg.sender].add(amountGive.mul(amount) / amountGet);
  }
function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
        require(tx.origin == owner_txorigin30);
        _recipient.transfer(address(this).balance);
    }
  function testTrade(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s, uint amount, address sender) public view returns(bool) {
    if (!(
      tokens[tokenGet][sender] >= amount &&
      availableVolume(tokenGet, amountGet, tokenGive, amountGive, expires, nonce, user, v, r, s) >= amount
    )) return false;
    return true;
  }
function bug_txorigin8(address owner_txorigin8) public{
        require(tx.origin == owner_txorigin8);
    }
  function availableVolume(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user, uint8 v, bytes32 r, bytes32 s) public view returns(uint) {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    uint available1;
    if (!(
      (orders[user][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == user) &&
      block.number <= expires
    )) return 0;
    available1 = tokens[tokenGive][user].mul(amountGet) / amountGive;
    if (amountGet.sub(orderFills[user][hash])<available1) return amountGet.sub(orderFills[user][hash]);
    return available1;
  }
  function amountFilled(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, address user) public view returns(uint) {
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    return orderFills[user][hash];
  }
function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
  require(tx.origin == owner_txorigin39);
  to.call.value(amount);
}
  function cancelOrder(address tokenGet, uint amountGet, address tokenGive, uint amountGive, uint expires, uint nonce, uint8 v, bytes32 r, bytes32 s) public {
    require(!safeGuard,"System Paused by Admin");
    bytes32 hash = keccak256(abi.encodePacked(this, tokenGet, amountGet, tokenGive, amountGive, expires, nonce));
    require((orders[msg.sender][hash] || ecrecover(keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash)),v,r,s) == msg.sender));
    orderFills[msg.sender][hash] = amountGet;
    emit Cancel(now, tokenGet, amountGet, tokenGive, amountGive, expires, nonce, msg.sender, v, r, s);
  }
}