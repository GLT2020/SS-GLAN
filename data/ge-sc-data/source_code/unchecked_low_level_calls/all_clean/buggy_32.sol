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
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success);
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool success);
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    }
contract ERC20Token is ERC20TokenInterface {
    using SafeMath for uint256;
  function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public totalSupply;
  function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) balances; 
  function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) allowed; 
    function balanceOf(address _owner) public view returns (uint256 value) {
        return balances[_owner];
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success) {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        emit Transfer(_from, _to, _value);
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract AsseteGram is ERC20Token {
  function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
  string public name = 'Electronic Gram';
  function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals = 3;
  function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol = 'eGram';
  function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  string public version = '2';
    constructor() public {
        address initialOwner = 0xac775cD446889ac167da466692449ece5439fc12;
        totalSupply = 180000000 * (10**uint256(decimals)); 
        balances[initialOwner] = totalSupply;
        emit Transfer(address(0), initialOwner, balances[initialOwner]);
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
    function() external {
        revert();
    }
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
}