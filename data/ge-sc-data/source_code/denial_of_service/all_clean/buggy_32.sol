pragma solidity ^0.5.7;
library SafeMath {
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
}
contract ERC20TokenInterface {
    function balanceOf(address _owner) public view returns (uint256 value);
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function transfer(address _to, uint256 _value) public returns (bool success);
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function approve(address _spender, uint256 _value) public returns (bool success);
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    }
contract ERC20Token is ERC20TokenInterface {
    using SafeMath for uint256;
  function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
  uint256 public totalSupply;
  bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
  mapping (address => uint256) balances; 
  bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
  mapping (address => mapping (address => uint256)) allowed; 
    function balanceOf(address _owner) public view returns (uint256 value) {
        return balances[_owner];
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function transfer(address _to, uint256 _value) public returns (bool success) {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
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
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
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
    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract AsseteGram is ERC20Token {
  function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
  string public name = 'Electronic Gram';
  function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
  uint8 public decimals = 3;
  function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
  string public symbol = 'eGram';
  function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
  string public version = '2';
    constructor() public {
        address initialOwner = 0xac775cD446889ac167da466692449ece5439fc12;
        totalSupply = 180000000 * (10**uint256(decimals)); 
        balances[initialOwner] = totalSupply;
        emit Transfer(address(0), initialOwner, balances[initialOwner]);
    }
function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
    function() external {
        revert();
    }
function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
}