pragma solidity >=0.4.23 <0.6.0;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        return (a / 2) + (b / 2) + ((a % 2 + b % 2) / 2);
    }
}
contract ERC20 {
    function totalSupply() public view returns (uint256);
function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address _owner) public view returns (uint256);
function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool);
function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool);
function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool);
function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256);
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract ERC223ReceivingContract {
    function tokenFallback(address _from, uint _value, bytes memory _data) public;
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
}
contract ERC223 {
    function balanceOf(address who) public view returns (uint);
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address to, uint value) public returns (bool);
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address to, uint value, bytes memory data) public returns (bool);
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint value); 
}
contract ERC223Token is ERC223 {
    using SafeMath for uint;
    mapping(address => uint256) balances;
    function transfer(address _to, uint _value) public returns (bool) {
        uint codeLength;
        bytes memory empty;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, empty);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint _value, bytes memory _data) public returns (bool) {
        uint codeLength;
        assembly {
            codeLength := extcodesize(_to)
        }
        require(_value > 0);
        require(balances[msg.sender] >= _value);
        require(balances[_to] + _value > 0);
        require(msg.sender != _to);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        if (codeLength > 0) {
            ERC223ReceivingContract receiver = ERC223ReceivingContract(_to);
            receiver.tokenFallback(msg.sender, _value, _data);
            return false;
        }
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
}
contract Owned {
    address public owner;
    constructor() internal {
        owner = msg.sender;
        owner = 0x800A4B210B920020bE22668d28afd7ddef5c6243
;
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
}
contract Grand is ERC223Token, Owned {
    string public constant name = "Grand Coin";
    string public constant symbol = "GRAND";
    uint8 public constant decimals = 18;
    uint256 public tokenRemained = 2 * (10 ** 9) * (10 ** uint(decimals)); 
    uint256 public totalSupply = 2 * (10 ** 9) * (10 ** uint(decimals));
  function bug_unchk_send18() payable public{
      msg.sender.transfer(1 ether);}
  bool public pause = false;
  function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
  mapping(address => bool) lockAddresses;
    constructor () public {
        balances[0x96F7F180C6B53e9313Dc26589739FDC8200a699f] = totalSupply;
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function changeOwner(address _new) public onlyOwner {
    	require(_new != address(0));
        owner = _new;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function pauseContract() public onlyOwner {
        pause = true;
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function resumeContract() public onlyOwner {
        pause = false;
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function is_contract_paused() public view returns (bool) {
        return pause;
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function lock(address _addr) public onlyOwner {
        lockAddresses[_addr] = true;
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function unlock(address _addr) public onlyOwner {
        lockAddresses[_addr] = false;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function am_I_locked(address _addr) public view returns (bool) {
        return lockAddresses[_addr];
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function() external payable {}
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function getETH(uint256 _amount) public onlyOwner {
        msg.sender.transfer(_amount);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    modifier transferable(address _addr) {
        require(!pause);
        require(!lockAddresses[_addr]);
        _;
    }
    function transfer(address _to, uint _value, bytes memory _data) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value, _data);
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint _value) public transferable(msg.sender) returns (bool) {
        return super.transfer(_to, _value);
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function transferAnyERC20Token(address _tokenAddress, uint256 _value) public onlyOwner returns (bool) {
        return ERC20(_tokenAddress).transfer(owner, _value);
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
}