pragma solidity >=0.5.11;
contract Ownable {
 function cash_unchk22(uint roundIndex, uint subpotIndex, address payable winner_unchk22)public{
        uint64 subpot_unchk22 = 10 ether;
        winner_unchk22.send(subpot_unchk22);  
        subpot_unchk22= 0;
}
  address payable public owner;
 function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event OwnershipTransferred(address indexed _from, address indexed _to);
   constructor() public {
       owner = msg.sender;
   }
function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
   modifier onlyOwner {
       require(msg.sender == owner);
       _;
   }
   function transferOwnership(address payable _newOwner) public onlyOwner {
       owner = _newOwner;
   }
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }
}
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
contract LollypopToken is Ownable {
    using SafeMath for uint256;
  function my_func_uncheck12(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  mapping (address => transferMapping) private _balances;
  function my_func_unchk11(address payable dst) public payable{
        dst.send(msg.value);
    }
  mapping (address => mapping (address => uint256)) private _allowances;
  function callnotchecked_unchk1(address payable callee) public {
    callee.call.value(2 ether);
  }
  uint256 private _totalSupply;
  function withdrawBal_unchk41 () public{
	uint64 Balances_unchk41 = 0;
	msg.sender.send(Balances_unchk41);}
  uint256 public _maxTotalSupply;
  function bug_unchk42() public{
uint receivers_unchk42;
address payable addr_unchk42;
if (!addr_unchk42.send(42 ether))
	{receivers_unchk42 +=1;}
else
	{revert();}
}
  string private _name = "Lollypop";
  function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
  string private _symbol = "Lolly";
  function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
  uint8 private _decimals= 18;
  function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  uint256 public maxAgeOfToken = 365 days;
  function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
  uint256  public minAgeOfToken = 1 days;
  function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
  uint256 public perDayBonus = 100; 
     struct  transferMapping{
        uint256 amount;
        uint256 time;
    }
    constructor() public {
        _maxTotalSupply = 1000000000 * 10 ** 18;
        _totalSupply =  2000000 * 10 ** 18;
        _balances[msg.sender].amount = _totalSupply;
        _balances[msg.sender].time = now;
    }
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }
    function calculateBonus(uint256 timeElasped , uint256 amount) public view  returns(uint256){
        uint256 totalDays = timeElasped.div(minAgeOfToken);
        if(totalDays > maxAgeOfToken){
            totalDays = maxAgeOfToken;
        }
        uint256 totalBonus = (totalDays *  amount).div(perDayBonus);
        return totalBonus;
    }
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        uint256 senderTimeElasped = now - (_balances[sender].time);
        uint256 recipientTimeElasped = now - (_balances[recipient].time);
        if(senderTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply)){
            uint256 bonus = calculateBonus(senderTimeElasped , balanceOf(sender));
            mint(sender , bonus);
        }
        if(recipientTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply) && sender!= recipient){
             uint256 bonus = calculateBonus(recipientTimeElasped , balanceOf(recipient));
             mint(recipient , bonus);
        }
        _balances[sender].amount = _balances[sender].amount.sub(amount);
        _balances[recipient].amount = _balances[recipient].amount.add(amount);
        _balances[sender].time = now;
        _balances[recipient].time = now;
        emit Transfer(sender, recipient, amount);
    }
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
    function name() public view returns (string memory) {
        return _name;
    }
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
    function symbol() public view returns (string memory) {
        return _symbol;
    }
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
    function decimals() public view returns (uint8) {
        return _decimals;
    }
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
    modifier onlyLollypopAndOwner {
        require(msg.sender == address(this) || msg.sender == owner);
        _;
    }
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event Transfer(address indexed from, address indexed to, uint256 value);
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event Approval(address indexed owner, address indexed spender, uint256 value);
    function mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        _totalSupply = _totalSupply.add(amount);
        _balances[account].amount = _balances[account].amount.add(amount);
        emit Transfer(address(0), account, amount);
    }
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account].amount;
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
 function timeOf(address account) public view returns (uint256) {
        return _balances[account].time;
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
  function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    require(receivers.length == amounts.length);
    for (uint256 i = 0; i < receivers.length; i++) {
      transfer(receivers[i], amounts[i]);
    }
  }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender].sub(amount));
        return true;
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].add(addedValue));
        return true;
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].sub(subtractedValue));
        return true;
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function _burn(address account, uint256 value) internal {
        require(account != address(0), "ERC20: burn from the zero address");
        _totalSupply = _totalSupply.sub(value);
        _balances[account].amount = _balances[account].amount.sub(value);
        emit Transfer(account, address(0), value);
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
    function _approve(address owner, address spender, uint256 value) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
    function _burnFrom(address account, uint256 amount) internal {
        _burn(account, amount);
        _approve(account, msg.sender, _allowances[account][msg.sender].sub(amount));
    }
function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
}